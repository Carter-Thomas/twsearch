use std::fmt::Debug;

use cubing::{alg::Move, kpuzzle::InvalidAlgError};

use crate::_internal::search::move_count::MoveCount;

// TODO: split this into 3 related traits.
/// The `Clone` implementation must be cheap for both the main struct as well as the `Pattern` and `Transformation` types (e.g. implemented using data shared with an `Arc` under the hood whenever any non-trivial amount of data is associated).
pub trait SemiGroupActionPuzzle: Debug + Clone {
    type Pattern: Eq + Clone + Debug;
    /// This is a proper "transformation" (such as a permutation) in the general
    /// case, but for `GenericPuzzleCore` it can be anything that is applied to a
    /// pattern, such as:
    ///
    /// - A [`Move`]
    /// - An index or reference into an array that encodes how to apply it
    type Transformation: Eq + Clone + Debug;

    // /********* Functions "defined on the puzzle". ********/
    // fn puzzle_default_pattern(&self) -> Self::Pattern;

    /********* Functions "defined on the move". ********/

    fn move_order(&self, r#move: &Move) -> Result<MoveCount, InvalidAlgError>;

    fn puzzle_transformation_from_move(
        &self,
        r#move: &Move,
    ) -> Result<Self::Transformation, InvalidAlgError>;

    // TODO: this is a leaky abstraction. use traits and enums to create a natural API for this.
    // TODO: `InvalidMoveError`?
    fn do_moves_commute(&self, move1: &Move, move2: &Move) -> Result<bool, InvalidAlgError>;

    /********* Functions "defined on the pattern". ********/

    fn pattern_apply_transformation(
        &self,
        pattern: &Self::Pattern,
        transformation_to_apply: &Self::Transformation,
    ) -> Option<Self::Pattern>;

    /// Must return `true`/`false` depending on whether
    /// `.pattern_apply_transformation(…)` would return `Some(…)`/`None`.
    ///
    /// If the return value is `false`, `into_pattern` may be left in a mangled
    /// state. The implementation must accept `into_pattern` values as input that
    /// were left mangled by a previous call, without affecting semantics.
    fn pattern_apply_transformation_into(
        &self,
        pattern: &Self::Pattern,
        transformation_to_apply: &Self::Transformation,
        into_pattern: &mut Self::Pattern,
    ) -> bool;
}

pub trait GroupActionPuzzle: SemiGroupActionPuzzle {
    // /********* Functions "defined on the puzzle". ********/
    // fn puzzle_transformation_from_alg(
    //     &self,
    //     alg: &Alg,
    // ) -> Result<Self::Transformation, InvalidAlgError>;
    // fn puzzle_identity_transformation(&self) -> Self::Transformation; // TODO: also define this on `KPuzzle` itself.
    //                                                                   // TODO: should this return owned `Move`s?
    fn puzzle_definition_all_moves(&self) -> Vec<Move>;

    // /********* Functions "defined on the transformation". ********/
    // // fn transformation_puzzle(transformation: &Self::Transformation) -> &Self; // TODO: add an additional trait for this.
    // fn transformation_invert(transformation: &Self::Transformation) -> Self::Transformation;
    // fn transformation_apply_transformation(
    //     transformation: &Self::Transformation,
    //     transformation_to_apply: &Self::Transformation,
    // ) -> Self::Transformation;
    // fn transformation_apply_transformation_into(
    //     transformation: &Self::Transformation,
    //     transformation_to_apply: &Self::Transformation,
    //     into_transformation: &mut Self::Transformation,
    // );
    // fn transformation_hash_u64(&self, transformation: &Self::Transformation) -> u64;
    // // TODO: efficient `order` function?
}

// This would be named `DefaultPattern`, but having the prefix "default" could suggest a relationship to the `Default` trait (which is not intended).
pub trait HasDefaultPattern: SemiGroupActionPuzzle {
    // /********* Functions "defined on the puzzle". ********/
    // TODO: Make this return a reference once `KPuzzle::default_pattern(…)` does.
    fn puzzle_default_pattern(&self) -> Self::Pattern;
}

pub trait HashablePatternPuzzle: SemiGroupActionPuzzle {
    fn pattern_hash_u64(&self, pattern: &Self::Pattern) -> u64;
}
