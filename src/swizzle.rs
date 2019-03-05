use std::mem;
use std::ptr;

/// A marker trait.
///
/// Implementing this trait can cause things to break in horrifying ways,
/// so please don't.
pub unsafe trait SZ {
    const N: usize;
}

/// A marker trait.
///
/// Implementing this trait can cause things to break in horrifying ways,
/// so please don't.
pub unsafe trait Not<T: SZ> {}

/// The swizzle trait is implemented for all types that can be swizzled.
///
/// In general, there should be no reason to implement this trait, but
/// doing so won't cause anything to break.
pub trait Swizzle<Mask> {
    type Output;

    fn swizzle(self, mask: Mask) -> Self::Output;
}

/// SwizzleInPlace is used for types where swizzling does not change the type
/// of the output.
///
/// Since every element in an array has the same type, it is possible to
/// rearange the array without consuming the array. This also has the advantage
/// that it is possible to swizzle slices.
pub trait SwizzleInPlace<Mask> {
    fn swizzle(&mut self, mask: Mask);
}

macro_rules! swizzle_types {
	($($t:ident),*;$($const:ident),*;$($num:expr),*;$($trait:ident),*) => (
		$(
			#[derive(Clone, Copy)]
			pub struct $t {}

			unsafe impl SZ for $t {
				const N: usize = $num;
			}

			pub const $const: $t = $t {};
		)*
		swizzle_not!(;$($t,)*;$($trait,)*);
		impl_swizzle!(;;$($t,)*;$($trait,)*);
	)
	}

macro_rules! impl_swizzle {
	($($types:ident,)*;$($used:ident,)*;;) => ();
	($($types:ident,)*;$($used:ident,)*;$t0:ident, $($t:ident,)*;$top:ident, $($trait:ident,)*) => (
		swizzle_where!($($types,)*$t0;$($used,)*T;$top);

		impl_swizzle!($($types,)*$t0,;$($used,)*$top,;$($t,)*;$($trait,)*);
	)
}

macro_rules! swizzle_where {
	($($mask:ident,)*;$($types:ident,)*;$($($t:ident,)*|)*;$($accum:ident,)*;;
		$trait:ident;$($($ta:ident,)*|)*) => (
		impl<$($mask,)*$($types),*> Swizzle<($($mask,)*)> for ($($types,)*)
		where $(
			$mask: SZ + $trait<$($ta),*> $(+ Not<$t>)*,
		)* {
			type Output = ($($mask::S,)*);

			fn swizzle(self, _: ($($mask,)*)) -> Self::Output {
				// this is incredibly unsafe because it works for non-copy types
				// to avoid possible double drop erros, the type validation must
				// must work correctly. This could theoretically fail if a another
				// crate were to implement a trait (for example `Not<A> for A`).
				// these traits are marked as unsafe, so this isn't a safety violation
				let out = unsafe {
					($($mask::from_tuple(ptr::read(&self)),)*)
				};
				// the elements of `self` have logically been moved into `out`
				mem::forget(self);
				out
			}
		}

		impl<$($mask,)*> Swizzle<($($mask,)*)> for usize
		where $(
			$mask: SZ,
		)* {
			type Output = usize;

			fn swizzle(self, _: ($($mask,)*)) -> usize {
				[$($mask::N),*][self]
			}
		}

		impl<$($mask,)*T> SwizzleInPlace<($($mask,)*)> for [T]
		where $(
			$mask: SZ $(+ Not<$t>)*,
		)* {
			fn swizzle(&mut self, _: ($($mask,)*)) {
				// neccesary to do this check before copying. Otherwise panics will
				// cause double-free errors on unwinding.
				if self.len() != num!(0, $($mask,)*) {
					panic!("Wrong sized slice for mask.");
				}
				let mut temp = unsafe {[$(ptr::read(&mut self[$mask::N])),*]};
				self.swap_with_slice(&mut temp);
				// temp contains carbon copies of the elements in self.
				mem::forget(temp);
			}
		}
	);
	($($mask:ident,)*;$($types:ident,)*;$($($t:ident,)*|)*;$($accum:ident,)*;
		$b0:ident, $($build:ident,)*;$trait:ident;$($($ta:ident,)*|)*) => (
		swizzle_where!($($mask,)*;$($types,)*;$($($t,)*|)*$($accum,)*|;$($accum,)*$b0,
			;$($build,)*;$trait;$($($ta,)*|)*$($types,)*|);
	);
	($($mask:ident),*;$($types:ident),*;$trait:ident) => (
		swizzle_where!($($mask,)*;$($types,)*;;;$($mask,)*;$trait;);
	)
}

macro_rules! num {
	($num:expr,) => (
		$num
	);
	($num:expr, $c:ident, $($count:ident,)*) => (
		num!($num + 1,$($count,)*)
	)
}

macro_rules! swizzle_not {
	($($start:ident,)*;;) => ();
	($($start:ident,)*;$t:ident, $($end:ident,)*;$trait:ident, $($traits:ident,)*) => (
		$(
			unsafe impl Not<$start> for $t {}
		)*
		$(
			unsafe impl Not<$end> for $t {}
		)*

		pub unsafe trait $trait<$($start,)*$t>: SZ {
			type S;

			#[doc="Returns on value from the tuple and 'forget's the rest."]
			fn from_tuple(tup: ($($start,)*$t,)) -> Self::S;
		}

		swizzle_traits!(;$($start,)*$t,;$trait);

		swizzle_not!($($start,)*$t,;$($end,)*;$($traits,)*);
	)
}

macro_rules! swizzle_traits {
	($($start:ident,)*;;$trait:ident) => ();
	($($start:ident,)*;$t:ident, $($types:ident,)*;$trait:ident) => (
		// neccesary to use a dummy type T to avoid naming conflicts
		unsafe impl<$($start,)*T,$($types),*> $trait<$($start,)*T,$($types),*> for $t {
			type S = T;

			fn from_tuple(tup: ($($start,)*T,$($types),*)) -> Self::S {
				let ($($start,)*t,$($types),*) = tup;
				$(
					mem::forget($start);
				)*
				$(
					mem::forget($types);
				)*
				t
			}
		}
		swizzle_traits!($($start,)*$t,;$($types,)*;$trait);
	)
}

swizzle_types!(R0, R1, R2, R3, R4, R5;S0, S1, S2, S3, S4, S5;0, 1, 2, 3, 4, 5;
	Swizzle0, Swizzle1, Swizzle2, Swizzle3, Swizzle4, Swizzle5);

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
tuple_vec_operations!(U1, U2, U3, U4, U5, U6, U7, U8,);
