use std::{collections::HashMap, mem, vec};

use cubing::kpuzzle::{KPattern, KPuzzle, KTransformation};
use thousands::Separable;

use crate::_internal::{
    canonical_fsm::{
        canonical_fsm::{CanonicalFSM, CanonicalFSMState, CANONICAL_FSM_START_STATE},
        search_generators::{FlatMoveIndex, SearchGenerators},
    },
    cli::args::{Generators, MetricEnum},
    errors::SearchError,
    gods_algorithm::factor_number::factor_number,
    search::indexed_vec::IndexedVec,
};

type SearchDepth = usize;

use indicatif::{MultiProgress, ProgressBar, ProgressFinish, ProgressStyle};

use super::bulk_queue::BulkQueue;

pub struct GodsAlgorithmTable {
    completed: bool, // "completed" instead of "complete" to make an unambiguous adjective
    pub pattern_to_depth: HashMap<KPattern, /* depth */ SearchDepth>,
}

impl GodsAlgorithmTable {
    pub fn new() -> Self {
        Self {
            completed: false,
            pattern_to_depth: HashMap::new(),
        }
    }
}

impl Default for GodsAlgorithmTable {
    fn default() -> Self {
        Self::new()
    }
}

struct QueueItem {
    canonical_fsm_state: CanonicalFSMState,
    pattern: KPattern,
}

pub struct GodsAlgorithmSearch {
    // params
    kpuzzle: KPuzzle,
    start_pattern: Option<KPattern>,
    search_generators: SearchGenerators<KPuzzle>,

    // TODO: find more elegant way to store this.
    cached_inverses: IndexedVec<FlatMoveIndex, KTransformation>,
    // state
    canonical_fsm: CanonicalFSM<KPuzzle>,
    pub(crate) table: GodsAlgorithmTable,
    bulk_queues: Vec<BulkQueue<QueueItem>>, // TODO: `HashMap` instead of `Vec` for the other layer for sparse rep?

    multi_progress_bar: MultiProgress,
}

macro_rules! format_num {
    ($n:expr) => {
        $n.separate_with_underscores()
    };
}

impl GodsAlgorithmSearch {
    pub fn try_new(
        kpuzzle: KPuzzle,
        start_pattern: Option<KPattern>,
        generators: &Generators,
        quantum_metric: &MetricEnum,
    ) -> Result<Self, SearchError> {
        let depth_to_patterns = vec![];
        let search_generators = SearchGenerators::try_new(
            &kpuzzle,
            generators.enumerate_moves_for_kpuzzle(&kpuzzle),
            quantum_metric,
            false,
        )?;
        let canonical_fsm = CanonicalFSM::new(
            kpuzzle.clone(),
            search_generators.clone(),
            Default::default(),
        );

        let cached_inverses = IndexedVec::new(
            search_generators
                .flat
                .iter()
                .map(|(flat_move_index, info)| {
                    if info.flat_move_index != flat_move_index {
                        panic!("Flat move indexing is out of order");
                    }
                    info.transformation.invert()
                })
                .collect(),
        );

        Ok(Self {
            kpuzzle,
            start_pattern,
            search_generators,
            canonical_fsm,
            table: GodsAlgorithmTable {
                completed: false,
                pattern_to_depth: HashMap::new(),
            },
            bulk_queues: depth_to_patterns,
            cached_inverses,
            multi_progress_bar: MultiProgress::new(),
        })
    }

    pub fn fill(&mut self) {
        let start_pattern = match &self.start_pattern {
            Some(start_pattern) => start_pattern.clone(),
            None => self.kpuzzle.default_pattern(),
        };
        self.table.pattern_to_depth.insert(start_pattern.clone(), 0);
        let start_item = QueueItem {
            canonical_fsm_state: CANONICAL_FSM_START_STATE,
            pattern: start_pattern,
        };
        self.bulk_queues.push(BulkQueue::new(Some(start_item)));

        let mut current_depth = 0;
        let mut num_patterns_total = 1;

        let start_time = instant::Instant::now();
        while !self.table.completed {
            let last_depth_patterns: BulkQueue<QueueItem> = mem::replace(
                &mut self.bulk_queues[current_depth],
                BulkQueue::bogus_new(), // TODO: change the field to avoid the need for this?
            );
            let num_last_depth_patterns = last_depth_patterns.size();

            current_depth += 1;

            let progress_bar = ProgressBar::new(num_last_depth_patterns.try_into().unwrap());
            let progress_bar = self.multi_progress_bar.insert_from_back(0, progress_bar);
            let progress_bar = progress_bar.with_finish(ProgressFinish::AndLeave);
            // TODO share the progress bar style?
            let progress_bar_style = ProgressStyle::with_template(
                "{prefix:3} {bar:12.cyan/blue} {elapsed:.2} {wide_msg}",
            )
            .expect("Could not construct progress bar.");
            // .progress_chars("=> ");
            progress_bar.set_style(progress_bar_style);
            progress_bar.set_prefix(current_depth.to_string());

            let num_to_test_at_current_depth: usize =
                num_last_depth_patterns * self.search_generators.flat.len();
            let mut num_tested_at_current_depth = 0;
            let mut patterns_at_current_depth = BulkQueue::new(None);
            for queue_item in last_depth_patterns.into_iter() {
                for move_class_index in self.search_generators.by_move_class.index_iter() {
                    let moves_in_class = self.search_generators.by_move_class.at(move_class_index);
                    let next_state = self
                        .canonical_fsm
                        .next_state(queue_item.canonical_fsm_state, move_class_index);
                    let next_state = match next_state {
                        Some(next_state) => next_state,
                        None => {
                            num_tested_at_current_depth += moves_in_class.len();
                            continue;
                        }
                    };
                    for move_info in moves_in_class {
                        num_tested_at_current_depth += 1;
                        let new_pattern = queue_item.pattern.apply_transformation(
                            self.cached_inverses.at(move_info.flat_move_index),
                        );
                        if self.table.pattern_to_depth.contains_key(&new_pattern) {
                            continue;
                        }

                        let new_item = QueueItem {
                            canonical_fsm_state: next_state,
                            pattern: new_pattern.clone(),
                        };
                        patterns_at_current_depth.push(new_item);
                        self.table
                            .pattern_to_depth
                            .insert(new_pattern, current_depth);

                        if num_tested_at_current_depth % 1000 == 0 {
                            progress_bar
                                .set_length(num_to_test_at_current_depth.try_into().unwrap());
                            progress_bar.set_position(num_tested_at_current_depth as u64);
                            progress_bar.set_message(format!(
                                "{} patterns ({} cumulative) — {} remaining candidates",
                                format_num!(patterns_at_current_depth.size()),
                                format_num!(num_patterns_total + patterns_at_current_depth.size()), // TODO: increment before
                                format_num!(
                                    num_to_test_at_current_depth
                                        - std::convert::TryInto::<usize>::try_into(
                                            num_tested_at_current_depth
                                        )
                                        .unwrap()
                                )
                            ))
                        }
                    }
                }
            }
            let num_patterns_at_current_depth = patterns_at_current_depth.size();
            num_patterns_total += num_patterns_at_current_depth;
            {
                progress_bar.set_length(patterns_at_current_depth.size().try_into().unwrap());
                progress_bar.set_position(patterns_at_current_depth.size().try_into().unwrap());
                progress_bar.set_message(format!(
                    "{} patterns ({} cumulative)",
                    format_num!(num_patterns_at_current_depth),
                    format_num!(num_patterns_total)
                ))
            }
            self.bulk_queues.push(patterns_at_current_depth);

            if num_patterns_at_current_depth == 0 {
                progress_bar.finish_and_clear();
                self.table.completed = true;
            } else {
                progress_bar.finish();
            }
        }
        let max_depth = current_depth - 1;
        println!();
        println!();
        println!(
            "Found {} ({}) pattern{}.\nMaximum depth: {} moves\nTotal time elapsed: {:?}",
            format_num!(num_patterns_total),
            factor_number(num_patterns_total.try_into().unwrap()),
            if num_patterns_total == 1 { "" } else { "s" },
            max_depth,
            instant::Instant::now() - start_time
        );
    }
}
