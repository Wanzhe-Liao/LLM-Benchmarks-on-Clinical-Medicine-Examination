# Figure Legends (Publication Export)

All figures report evaluation on the CMCA 165-item benchmark (Clinical Medicine Comprehensive Ability; December 21, 2025 exam) across 11 models under two protocols: **Baseline** (direct prompting) and **Agentic** (Planner-Reasoner-Critic). Unless otherwise stated, each model-protocol pair was evaluated in 5 independent trials.

## Fig. 1. Baseline benchmark profile (direct prompting).

(A) Overall accuracy by model (mean +/- 95% CI across 5 trials; 165 items per trial). Bars are colored by model availability (proprietary vs open-weight), and models are ordered by mean accuracy.  
(B) Model-item stability map under the baseline protocol. Each cell summarizes the number of correct trials for a given model-question pair (0-5) using four discrete categories: robust correct (5/5), probabilistic (3-4/5), guessing/ambiguous (1-2/5), and robust wrong (0/5). Questions are grouped by discipline and sorted within each discipline from easier to harder based on the across-model mean correct count; models are ordered identically to Panel A. Vertical white separators denote discipline blocks.

## Fig. 2. Agentic benchmark profile (Planner-Reasoner-Critic workflow).

(A) Overall accuracy by model (mean +/- 95% CI across 5 trials; 165 items per trial) under the agentic protocol; colors and ordering follow Fig. 1A.  
(B) Model-item stability map under the agentic protocol, using the same four-category encoding as Fig. 1B. Questions are grouped by discipline and sorted within each discipline from easier to harder based on the across-model mean correct count under the agentic protocol; models are ordered identically to Panel A.

## Fig. 3. Overall performance change from baseline to agentic evaluation.

For each model, mean accuracy (+/- 95% CI across 5 trials) is shown for baseline (gray) and agentic (blue) protocols. Right-side annotations report the paired difference (agentic - baseline) in percentage points. Models are ordered by baseline mean accuracy.

## Fig. 4. Agentic Iatrogenic Risk Analysis (AIRA): mechanism and risk stratification.

(A) Transition "butterfly" plot summarizing majority-vote correctness transitions from baseline to agentic evaluation, stratified by baseline item difficulty. For each model-item unit, majority correctness is defined as >=3/5 trials correct. Transitions are categorized as rescue (0->1), iatrogenic error (1->0), retained mastery (1->1), or refractory error (0->0); the plot displays rescue and iatrogenic counts with a net line (rescue - iatrogenic). Baseline difficulty bins are defined by the fraction of models that are majority-correct for an item under baseline (>=80%, 60-80%, 40-60%, 20-40%, <20%).  
(B) Model-level safety-efficacy trade-off. The Safety Index is the fraction of baseline-majority-correct items that remain majority-correct under agentic evaluation (C->C / Base_Correct). The Rescue Index is the fraction of baseline-majority-wrong items that become majority-correct under agentic evaluation (W->C / Base_Wrong). Bubble size is proportional to the damage-to-rescue ratio (DRR = iatrogenic / rescue), and a red outline indicates statistically significant net harm by McNemar's exact test with Benjamini-Hochberg FDR correction across models (q<0.05).

## Fig. 5. Baseline-defined cold-spot items and agentic "rescue" analysis.

Dumbbell plot of per-item overall accuracy for baseline-defined cold spots (baseline overall accuracy <20% across all model x trial responses). Each row corresponds to one cold-spot item (labeled by question ID, discipline, and question type), with baseline (gray) and agentic (blue) overall accuracies connected by a line; the dashed vertical line marks the 20% cutoff. Items are ordered by agentic improvement (delta accuracy, percentage points).

## Fig. 6. Rasch (1PL) Wright maps for baseline vs agentic protocols.

Rasch (1-parameter logistic) person-item maps showing the distributions of estimated model ability (theta; left histogram) and item difficulty (b; right histogram) on a shared logit scale. "Persons" correspond to model-by-trial instances (11 models x 5 trials), and items correspond to the 165 questions. Panel A: baseline; Panel B: agentic.

## Fig. 7. Classical test theory (CTT) item difficulty vs discrimination.

Scatter plots of item easiness (x-axis, % correct across model-by-trial persons) versus discrimination (y-axis, point-biserial correlation between the item score and the total score excluding that item). Point color encodes discrimination. Dashed horizontal reference lines mark r=0.2 and r=0.6. Panel A: baseline; Panel B: agentic.

## Fig. S1. Nemenyi critical-difference (CD) diagrams for baseline and agentic evaluations.

Average model ranks computed across the 165 items (lower is better), with the Nemenyi CD (alpha=0.05) shown as a reference bar. Models are colored by whether they are significantly worse than the best model (rank difference > CD). Panel A: baseline; Panel B: agentic.

## Fig. S2. Accuracy vs self-consistency across repeated trials.

Relationship between mean accuracy and Fleiss' kappa (agreement across the 5 repeated trials) for each model. Panel A: baseline; Panel B: agentic.

## Fig. S3. Discipline gap between basic sciences and clinical disciplines.

For each model, dumbbell arrows compare accuracy on basic sciences (Physiology/Biochemistry + Pathology; circle) versus clinical disciplines (Internal Medicine + Surgery; square). The right-aligned label reports the clinical-minus-basic gap in percentage points. Panel A: baseline; Panel B: agentic.

## Fig. S4. Entropy heatmaps of trial-to-trial output uncertainty.

Per model-item response uncertainty quantified as Shannon entropy (bits) over the distribution of selected options across 5 trials. Cells with zero entropy (identical outputs across trials) are masked in white. Items are grouped by discipline and sorted within each discipline by the across-model mean entropy; models are ordered by overall accuracy. Panel A: baseline; Panel B: agentic.

## Fig. S5. Consistency Sankey (model -> outcome stability -> discipline).

Sankey diagrams summarizing stability categories based on the number of correct trials for each model-item pair: robust correct (5/5), probabilistic (3-4/5), guessing/ambiguous (1-2/5), and robust wrong (0/5). Left flows show the per-model distribution across the 165 items; right flows summarize the overall discipline composition of each stability category aggregated across all models. Panel A: baseline; Panel B: agentic.

## Fig. S6. Latency-accuracy Pareto frontiers under baseline and agentic protocols.

Scatter plots of mean latency (log scale; lower is better) versus mean accuracy for each model. Marker shape encodes model availability (proprietary vs open-weight); bubble size reflects parameter count when available (open-weight models). The dashed curve indicates the Pareto frontier (non-dominated models). Panel A: baseline; Panel B: agentic.

## Fig. S7. UpSet plots of "robust wrong" (0/5 correct) intersections across models.

UpSet plots summarize intersections of items that were robust wrong (0/5 correct across trials) for different subsets of models. The bar chart reports the number of items in each intersection (top intersections ranked by size), and the dot matrix indicates which models are included in each subset (models ordered by overall accuracy). The all-models intersection, when present, is highlighted. Panel A: baseline; Panel B: agentic.
