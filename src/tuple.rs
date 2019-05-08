use std::mem;

/// A TupleIndex is a compile time type that represents the index of an element in a tuple.
/// Using this trait, the provided types can be used to pick elements from a tuple at compile
/// time.
///
/// It can safely be assumed that this trait will only be implemented for the types defined
/// in this module, and that T will always be a tuple type.
pub unsafe trait TupleIndex<T>: TupleNum {
    type I;

    /// Returns one element of the tuple and `forgets` the rest.
    fn get(t: T) -> Self::I;

    fn get_ref<'a>(t: &'a T) -> &'a Self::I;

    fn get_mut_ref<'a>(t: &'a mut T) -> &'a mut Self::I;
}

pub unsafe trait TupleNum {
    const N: usize;
}

macro_rules! tup_index {
	($t:ident;$($start:ident,)*;$u:ident;$($end:ident,)*) => (
		#[allow(unused)]
		unsafe impl<$($start,)*$u,$($end,)*> TupleIndex<($($end,)*$u,$($start,)*)> for $t {
			type I = $u;

			fn get(t: ($($end,)*$u,$($start,)*)) -> $u {
				let ($($end,)*$u,$($start,)*) = t;
				$(
					mem::forget($start);
				)*
				$(
					mem::forget($end);
				)*
				$u
			}

			fn get_ref<'a>(t: &'a ($($end,)*$u,$($start,)*)) -> &'a $u {
				let ($($end,)*$u,$($start,)*) = t;
				&$u
			}

			fn get_mut_ref<'a>(t: &'a mut ($($end,)*$u,$($start,)*)) -> &'a mut $u {
				let ($($end,)*ref mut$u,$($start,)*) = t;
				$u
			}
		}
	);
	(;$t:ident, $($ts:ident,)*;$($start:ident,)*;$u:ident;) => (
		tup_index!($t;$($start,)*;$u;);
	);
	(;$t:ident, $($ts:ident,)*;$($start:ident,)*;$u:ident;$e:ident,$($end:ident,)*) => (
		tup_index!($t;$($start,)*;$u;$e,$($end,)*);

		tup_index!(;$($ts,)*;$($start,)*$u,;$e;$($end,)*);
	);
}

macro_rules! impl_tuple {
	($n:expr; $($t:ident,)*;$($c:ident,)*;$($u:ident,)*) => (
		$(
			pub struct $t {}

			pub const $c: $t = $t {};
		)*

		impl_tuple!($n;$($t,)*;$($u,)*);
	);
	($n:expr;;) => ();
	($n:expr; $t0:ident,$($t:ident,)*;$u0:ident,$($u:ident,)*) => (
		unsafe impl TupleNum for $t0 {
			const N: usize = $n;
		}

		tup_index!(;$t0,$($t,)*;;$u0;$($u,)*);
		impl_tuple!(($n - 1); $($t,)*;$($u,)*);
	);
}

#[cfg(feature = "longer_tuples")]
impl_tuple!(
	16;
	T15, T14, T13, T12, T11, T10, T9, T8, T7, T6, T5, T4, T3, T2, T1, T0,;
	I15, I14, I13, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0,;
	U15, U14, U13, U12, U11, U10, U9, U8, U7, U6, U5, U4, U3, U2, U1, U0,
);

#[cfg(not(feature = "longer_tuples"))]
impl_tuple!(
	16;
	T15, T14, T13, T12, T11, T10, T9, T8, T7, T6, T5, T4, T3, T2, T1, T0,;
	I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0,;
	U15, U14, U13, U12, U11, U10, U9, U8, U7, U6, U5, U4, U3, U2, U1, U0,
);

pub trait AttachFront<T> {
    type AttachFront;

    fn attach_front(self, t: T) -> Self::AttachFront;
}

pub trait AttachBack<T> {
    type AttachBack;

    fn attach_back(self, t: T) -> Self::AttachBack;
}

pub trait RemoveFront {
    type Front;
    type Remaining;

    fn remove_front(self) -> (Self::Front, Self::Remaining);
}

pub trait RemoveBack {
    type Back;
    type Remaining;

    fn remove_back(self) -> (Self::Back, Self::Remaining);
}

macro_rules! tuple_vec_operations {
	() => ();
	($t0:ident, $($t:ident,)*) => (
		impl<$t0,$($t,)*> AttachFront<$t0> for ($($t,)*) {
			type AttachFront = ($t0,$($t,)*);

			fn attach_front(self, v: $t0) -> Self::AttachFront {
				let ($($t,)*) = self;

				(v,$($t,)*)
			}
		}

		impl<$t0,$($t,)*> AttachBack<$t0> for ($($t,)*) {
			type AttachBack = ($($t,)*$t0,);

			fn attach_back(self, v: $t0) -> Self::AttachBack {
				let ($($t,)*) = self;

				($($t,)*v,)
			}
		}

		impl<$t0,$($t,)*> RemoveFront for ($t0,$($t,)*) {
			type Front = $t0;
			type Remaining = ($($t,)*);

			fn remove_front(self) -> (Self::Front, Self::Remaining) {
				let ($t0,$($t,)*) = self;

				($t0, ($($t,)*))
			}
		}

		impl<$t0,$($t,)*> RemoveBack for ($($t,)*$t0,) {
			type Back = $t0;
			type Remaining = ($($t,)*);

			fn remove_back(self) -> (Self::Back, Self::Remaining) {
				let ($($t,)*$t0,) = self;

				($t0, ($($t,)*))
			}
		}

		tuple_vec_operations!($($t,)*);
	)
}

#[cfg(feature = "longer_tuples")]
tuple_vec_operations!(
    U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21,
    U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40,
    U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59,
    U60, U61, U62, U63, U64,
);

#[cfg(not(feature = "longer_tuples"))]
tuple_vec_operations!(U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16,);
