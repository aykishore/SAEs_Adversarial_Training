# CLIP representation engineering


The project reproduced and extends the visual representation-reading experiment from Representation Engineering: A Top-Down Approach to AI Transparency https://arxiv.org/abs/2310.01405

# Experiment

Model: openai/clip-vit-base-patch32

Dataset: ERG-DB, using the Mery character and six emotion-vs-neutral tasks.

The experiment extracts hidden representations from all 12 CLIP vision transformer blocks and learns Linear Artificial Tomography (LAT) directions using PCA over contrastive representation differences.

# analyses
CLIP hidden-state extraction across all 12 transformer blocks.
LAT/PCA representation reading
Emotion vs neutral evaluation
Random-pair LAT experiments
Cross-emotion generalization analysis
Cosine similarity analysis between emotion directions
CLS-token vs patch-token ablation 

# observation
Emotion-vs-neutral representations were highly separable on the sampled FERG-DB split. However, cross-emotion evaluation showed that several learned directions generalized strongly to other facial expressions.

For example, some directions that were geometrically distinct according to cosine similarity nevertheless produced similar functional classifications across emotions.

This suggests that geometric similarity between dense representation directions does not necessarily correspond directly to functional concept specificity.

# setup
install dependencies
FERG_DB by Deepali Aneja, Alex Colburn, Gary Faigin, Linda G. Shapiro, Barbara Mones, "Modeling Stylized Character Expressions via Deep Learning." Asian Conference on Computer Vision. Springer International Publishing, 2016.
