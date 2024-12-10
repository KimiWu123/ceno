use crate::{
    chip_handler::{AddressExpr, MemoryExpr, RegisterExpr},
    circuit_builder::CircuitBuilder,
    error::{UtilError, ZKVMError},
    expression::{Expression, ToExpr, WitIn},
    gadgets::{AssertLTConfig, SignedExtendConfig},
    instructions::riscv::constants::{LIMB_MASK, MAX_RANGE_CHECK, UInt},
    uint::util::max_carry_word_for_multiplication,
    utils::add_one_to_big_num,
    witness::LkMultiplicity,
};
use ark_std::iterable::Iterable;
use ff::Field;
use ff_ext::ExtensionField;
use goldilocks::SmallField;
use itertools::{Itertools, enumerate};
use std::{
    borrow::Cow,
    mem::{self, MaybeUninit},
    ops::Index,
    usize,
};
pub use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use sumcheck::util::ceil_log2;

pub trait AllowedTypes: Clone {
    fn from_u32(value: u32) -> Self;
    fn from_u64(value: u64) -> Self;
    fn into_u64(self) -> u64;
}

impl AllowedTypes for u32 {
    fn from_u32(value: u32) -> Self {
        value
    }

    fn from_u64(value: u64) -> Self {
        value as u32
    }

    fn into_u64(self) -> u64 {
        self as u64
    }
}

impl AllowedTypes for u64 {
    fn from_u32(value: u32) -> Self {
        value as u64
    }

    fn from_u64(value: u64) -> Self {
        value
    }

    fn into_u64(self) -> u64 {
        self
    }
}
/// A struct holding intermediate results of arithmetic add operations from Value
pub struct ValueAdd<T: AllowedTypes + Copy + Default> {
    pub limbs: Vec<T>,
    pub carries: Vec<T>,
}

/// A struct holding intermediate results of arithmetic mul operations from Value
pub struct ValueMul<T: AllowedTypes + Copy + Default, const C: usize> {
    pub limbs: Vec<T>,
    pub carries: Vec<T>,
    pub max_carry_value: T,
}

impl<T: AllowedTypes + Copy + Default, const C: usize> ValueMul<T, C> {
    pub fn as_hi_value(&self) -> Value<T, C> {
        Value::<T, C>::from_limb_slice_unchecked(self.as_hi_limb_slice())
    }

    pub fn as_hi_limb_slice(&self) -> &[T] {
        &self.limbs[self.limbs.len() / 2..]
    }
}

#[derive(Clone)]
// T: AllowedTypes + Copy + Default
pub struct Value<'a, T: AllowedTypes + Copy + Default, const C: usize> {
    val: u64,
    pub limbs: Cow<'a, [T]>,
}

// TODO generalize to support non 16 bit limbs
// TODO optimize api with fixed size array
impl<'a, T: AllowedTypes + Copy + Default, const C: usize> Value<'a, T, C> {
    const M: usize = { mem::size_of::<T>() * 8 };
    const LIMBS: usize = Self::M.div_ceil(C);

    pub fn new(val: u64, lkm: &mut LkMultiplicity) -> Self {
        let uint = Value::<T, C> {
            val,
            limbs: Cow::Owned(Self::split_to_ux(val)),
        };
        Self::assert_ux(&uint.limbs, lkm);
        uint
    }

    pub fn new_unchecked(val: u64) -> Self {
        Value::<T, C> {
            val,
            limbs: Cow::Owned(Self::split_to_ux(val)),
        }
    }

    pub fn from_limb_unchecked(limbs: Vec<T>) -> Self {
        Value::<T, C> {
            val: limbs
                .iter()
                .rev()
                .fold(0u64, |acc, &v| (acc << C) + v.into_u64())
                .into(),
            limbs: Cow::Owned(limbs),
        }
    }

    pub fn from_limb_slice_unchecked(limbs: &'a [T]) -> Self {
        Value::<T, C> {
            val: limbs
                .iter()
                .rev()
                .fold(0u64, |acc, &v| (acc << C) + v.into_u64())
                .into(),
            limbs: Cow::Borrowed(limbs),
        }
    }

    fn assert_ux(v: &[T], lkm: &mut LkMultiplicity) {
        v.iter().for_each(|v| {
            lkm.assert_ux::<C>((*v).into_u64());
        })
    }

    fn split_to_ux(value: u64) -> Vec<T> {
        let mask = (1 << C) - 1;
        (0..Self::LIMBS)
            .scan(value, |acc, _| {
                let limb = *acc & mask;
                *acc >>= C;
                Some(T::from_u64(limb))
            })
            .collect_vec()
    }

    pub fn as_limbs(&self) -> &[T] {
        &self.limbs
    }

    /// Convert the limbs to a u64 value
    pub fn as_u64(&self) -> u64 {
        self.val.into()
    }

    /// Convert the limbs to a u32 value
    pub fn as_u32(&self) -> u32 {
        self.as_u64() as u32
    }

    pub fn add(&self, rhs: &Self, lkm: &mut LkMultiplicity, with_overflow: bool) -> ValueAdd<T> {
        let res =
            self.as_limbs()
                .iter()
                .zip(rhs.as_limbs())
                .fold(vec![], |mut acc, (a_limb, b_limb)| {
                    let a_limb = a_limb.into_u64();
                    let b_limb = b_limb.into_u64();
                    let (a, b) = a_limb.overflowing_add(b_limb);
                    if let Some((_, prev_carry)) = acc.last() {
                        let (e, d) = a.overflowing_add(*prev_carry);
                        acc.push((e, (b || d) as u64));
                    } else {
                        acc.push((a, b as u64));
                    }
                    // range check
                    if let Some((limb, _)) = acc.last() {
                        lkm.assert_ux::<C>(*limb as u64);
                    };
                    acc
                });
        let (limbs, mut carries): (Vec<T>, Vec<T>) = res
            .into_iter()
            .map(|(v, c)| (T::from_u64(v), T::from_u64(c)))
            .unzip();

        if !with_overflow {
            carries.resize(carries.len() - 1, T::from_u64(0));
        }
        ValueAdd::<T> { limbs, carries }
    }

    pub fn mul(&self, rhs: &Self, lkm: &mut LkMultiplicity, with_overflow: bool) -> ValueMul<T, C> {
        self.internal_mul(rhs, lkm, with_overflow, false)
    }

    pub fn mul_hi(
        &self,
        rhs: &Self,
        lkm: &mut LkMultiplicity,
        with_overflow: bool,
    ) -> ValueMul<T, C> {
        self.internal_mul(rhs, lkm, with_overflow, true)
    }

    #[allow(clippy::type_complexity)]
    pub fn mul_add(
        &self,
        mul: &Self,
        addend: &Self,
        lkm: &mut LkMultiplicity,
        with_overflow: bool,
    ) -> (ValueAdd<T>, ValueMul<T, C>) {
        let mul_result = self.internal_mul(mul, lkm, with_overflow, false);
        let add_result = addend.add(
            &Self::from_limb_unchecked(mul_result.limbs.clone()),
            lkm,
            with_overflow,
        );
        (add_result, mul_result)
    }

    fn internal_mul(
        &self,
        mul: &Self,
        lkm: &mut LkMultiplicity,
        with_overflow: bool,
        with_hi_limbs: bool,
    ) -> ValueMul<T, C> {
        let a_limbs = self.as_limbs();
        let b_limbs = mul.as_limbs();

        let num_limbs = if !with_hi_limbs {
            a_limbs.len()
        } else {
            2 * a_limbs.len()
        };
        let mut c_limbs = vec![T::from_u64(0); num_limbs];
        let mut carries = vec![T::from_u64(0); num_limbs];
        let mut tmp = vec![0u64; num_limbs];
        enumerate(a_limbs).for_each(|(i, &a_limb)| {
            enumerate(b_limbs).for_each(|(j, &b_limb)| {
                let idx = i + j;
                if idx < num_limbs {
                    tmp[idx] += a_limb.into_u64() * b_limb.into_u64();
                }
            })
        });

        tmp.iter()
            .zip(c_limbs.iter_mut())
            .enumerate()
            .for_each(|(i, (tmp, limb))| {
                // tmp + prev_carry - carry * Self::LIMB_BASE_MUL
                let mut tmp = (*tmp).into_u64();
                if i > 0 {
                    tmp += carries[i - 1].into_u64();
                }
                // update carry
                carries[i] = T::from_u64(tmp >> C);
                // update limb with only lsb 16 bit
                *limb = T::from_u64(tmp);
            });

        if !with_overflow {
            // If the outcome overflows, `with_overflow` can't be false
            assert_eq!(
                carries[carries.len() - 1].into_u64(),
                0,
                "incorrect overflow flag"
            );
            carries.resize(carries.len() - 1, T::from_u64(0));
        }

        // range check
        c_limbs
            .iter()
            .for_each(|&c| lkm.assert_ux::<C>(c.into_u64()));

        ValueMul {
            limbs: c_limbs,
            carries,
            max_carry_value: T::from_u64(max_carry_word_for_multiplication(2, Self::M, C)),
        }
    }
}

#[cfg(test)]
mod tests {

    mod value {
        use crate::{Value, witness::LkMultiplicity};
        #[test]
        fn test_add() {
            let a = Value::new_unchecked(1u32);
            let b = Value::new_unchecked(2u32);
            let mut lkm = LkMultiplicity::default();

            let ret = a.add(&b, &mut lkm, true);
            assert_eq!(ret.limbs[0], 3);
            assert_eq!(ret.limbs[1], 0);
            assert_eq!(ret.carries[0], 0);
            assert_eq!(ret.carries[1], 0);
        }

        #[test]
        fn test_add_carry() {
            let a = Value::new_unchecked(u16::MAX as u32);
            let b = Value::new_unchecked(2u32);
            let mut lkm = LkMultiplicity::default();

            let ret = a.add(&b, &mut lkm, true);
            assert_eq!(ret.limbs[0], 1);
            assert_eq!(ret.limbs[1], 1);
            assert_eq!(ret.carries[0], 1);
            assert_eq!(ret.carries[1], 0);
        }

        #[test]
        fn test_mul() {
            let a = Value::new_unchecked(1u32);
            let b = Value::new_unchecked(2u32);
            let mut lkm = LkMultiplicity::default();

            let ret = a.mul(&b, &mut lkm, true);
            assert_eq!(ret.limbs[0], 2);
            assert_eq!(ret.limbs[1], 0);
            assert_eq!(ret.carries[0], 0);
            assert_eq!(ret.carries[1], 0);
        }

        #[test]
        fn test_mul_carry() {
            let a = Value::new_unchecked(u16::MAX as u32);
            let b = Value::new_unchecked(2u32);
            let mut lkm = LkMultiplicity::default();

            let ret = a.mul(&b, &mut lkm, true);
            assert_eq!(ret.limbs[0], u16::MAX - 1);
            assert_eq!(ret.limbs[1], 1);
            assert_eq!(ret.carries[0], 1);
            assert_eq!(ret.carries[1], 0);
        }

        #[test]
        fn test_mul_overflow() {
            let a = Value::new_unchecked(u32::MAX / 2 + 1);
            let b = Value::new_unchecked(2u32);
            let mut lkm = LkMultiplicity::default();

            let ret = a.mul(&b, &mut lkm, true);
            assert_eq!(ret.limbs[0], 0);
            assert_eq!(ret.limbs[1], 0);
            assert_eq!(ret.carries[0], 0);
            assert_eq!(ret.carries[1], 1);
        }
    }
}
