# -*- coding: utf-8 -*-
"""One-off script: assembles the results article as a .docx file from the
pipeline's actual output data. Run once, not part of the reusable pipeline."""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

def h1(text):
    doc.add_heading(text, level=1)

def h2(text):
    doc.add_heading(text, level=2)

def p(text):
    doc.add_paragraph(text)

FIG_DIR = r"D:\Desktop\endo-fibro-lit-mining\figures"

def figure(filename, caption, width_in=6.0):
    doc.add_picture(f"{FIG_DIR}\\{filename}", width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9.5)
    doc.add_paragraph()

def title(text):
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = h.add_run(text)
    run.bold = True
    run.font.size = Pt(16)

def subtitle(text):
    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = h.add_run(text)
    run.italic = True
    run.font.size = Pt(11)

title("A Validated NLP and Machine Learning Pipeline for the "
      "Endometriosis-Fibromyalgia Literature: Quantifying the Gap and "
      "Surfacing Latent Cross-Literature Connections")
subtitle("Higher Institute of Biotechnology of Sidi Thabet (ISBST), University of Manouba, Sidi Thabet, Tunisia")
doc.add_paragraph()

h1("Abstract")
p(
"A recent mini-review argued that natural language processing (NLP) and machine "
"learning (ML) methods, already applied separately to the endometriosis and "
"fibromyalgia literatures, have never been applied to the comorbidity between "
"the two conditions as a unit of analysis. Here we execute that proposal on "
"public PubMed literature, requiring no ethics approval since no human-subjects "
"data is used. We built a corpus-construction pipeline with a built-in "
"false-positive audit, an LLM-based structured claim extraction pipeline "
"validated against independent human annotation, a comparative topic-modeling "
"analysis of the joint versus single-disease literatures, and a retrieval-based "
"method for surfacing papers in the separate endometriosis and fibromyalgia "
"literatures that describe the same underlying mechanism without citing each "
"other. Of 91 papers that literally co-mention both conditions, only 16 (17.6%) "
"treat the comorbidity as their primary subject; the rest mention both only in "
"passing. Structured extraction, validated by a human reader against a "
"15-paper sample, achieved near-perfect agreement on mechanism classification "
"(Cohen's kappa = 0.89) and substantial agreement on study design (kappa = "
"0.75), with a documented, honestly-reported weakness in one finer-grained "
"field (kappa = 0.32). Most importantly, a retrieval-augmented discovery "
"pipeline identified 22 verified pairs of papers, one from the fibromyalgia "
"literature and one from the endometriosis literature, that describe the same "
"mechanism (most often central sensitization, nociplastic pain, or NLRP3 "
"inflammasome signaling) without citing each other. These results demonstrate "
"that the methodological gap identified in the original mini-review is real, "
"measurable, and addressable with entirely free, publicly available tools."
)

h1("1. Introduction")
p(
"Endometriosis and fibromyalgia are increasingly understood as linked through "
"shared central sensitization mechanisms, with cohort and comparative clinical "
"studies describing comorbid patients as a distinct, more severely affected "
"phenotype. A recent mini-review mapped how NLP, ML, and deep learning methods "
"are already used in adjacent problems, literature-scale topic modeling, "
"clinical named-entity recognition, EHR-based phenotyping, and early "
"multimodal ML, and argued that none of this literature performs joint "
"extraction across both conditions simultaneously: every existing pipeline is "
"built, trained, and validated for a single disease. This paper executes that "
"proposal directly, using PubMed literature alone. Restricting the analysis to "
"public, already-published abstracts (rather than electronic health records or "
"patient-generated social media text) was a deliberate methodological choice: "
"it avoids the human-subjects, consent, and re-identification concerns that "
"the original mini-review flagged as requiring a defensible data-governance "
"protocol, while still allowing a genuine test of whether NLP and ML methods "
"can surface comorbidity-relevant structure that a single-disease reading of "
"the literature would miss."
)

h1("2. Methods")

h2("2.1 Corpus construction and false-positive audit")
p(
"Three corpora were built via the NCBI PubMed E-utilities API: a joint corpus "
"(records matching both \"endometriosis\"[tiab] and \"fibromyalgia\"[tiab]), "
"an endometriosis-only corpus, and a fibromyalgia-only corpus, each restricted "
"to literal title/abstract co-mention. PubMed's search API imposes a hard cap "
"of 9,999 results per query regardless of pagination; corpora exceeding this "
"cap (the two single-disease corpora) were retrieved via recursive date-range "
"partitioning, splitting any date range whose result count exceeded the cap "
"until every partition fell under it, then de-duplicating across partitions. "
"A systematic false-positive audit was then run on every corpus: because "
"PubMed's automatic term-mapping can silently broaden a [tiab] query beyond "
"literal text (matching related MeSH concepts instead), each fetched record "
"was checked for the literal presence of its required disease term(s) in the "
"fetched title/abstract text. Records failing this check were excluded before "
"any further analysis."
)

h2("2.2 Structured claim extraction")
p(
"Each joint-corpus abstract was classified by a large language model (Gemini "
"2.5 Flash-Lite, free tier, structured JSON output) into six fields: the "
"mechanism proposed or investigated as linking the two conditions (nine "
"categories, including central sensitization, nociplastic pain, autoimmune, "
"hormonal, and neuropathic/small-fiber pathways, plus not-specified); study "
"design; sample size; comorbidity focus (whether the comorbidity is the "
"paper's primary subject, a secondary point, or an incidental co-mention); a "
"field distinguishing papers where fibromyalgia is a genuine diagnosed "
"comorbidity from papers that use a fibromyalgia-derived survey instrument "
"purely as a continuous nociplastic-pain measure in patients who may not have "
"fibromyalgia; and the paper's stance toward a shared-mechanism hypothesis "
"between the two conditions specifically, based only on what the paper's own "
"results and discussion claim, not on background framing alone."
)

h2("2.3 Human validation")
p(
"To assess extraction reliability, a random sample of joint-corpus papers was "
"independently annotated by a human reader (blind to the model's output) "
"using the same six-field schema, then compared against the model's "
"extraction using raw percent agreement and Cohen's kappa (chance-corrected "
"agreement) per field. One field (the fibromyalgia-measurement-type "
"distinction) showed weak initial agreement; the extraction prompt was then "
"revised, first by tightening the category definitions and then by adding "
"three concrete worked examples drawn from the actual disagreement cases, and "
"re-validated against the same human annotations."
)

h2("2.4 Comparative topic modeling")
p(
"Topic modeling was run on the joint corpus using two independent methods for "
"comparison: BERTopic, using TF-IDF-plus-truncated-SVD document embeddings "
"(a torch-free substitute for transformer sentence embeddings, adopted after "
"the evaluation environment's PyTorch installation proved unable to load, an "
"environment-level failure unrelated to this project's code) combined with "
"UMAP dimensionality reduction and HDBSCAN clustering; and classical Latent "
"Dirichlet Allocation (LDA) as a baseline, to report where the two methods "
"agree or diverge rather than asserting a single method's output as ground "
"truth."
)

h2("2.5 Retrieval-augmented cross-literature discovery")
p(
"The endometriosis-only and fibromyalgia-only corpora (with joint-corpus "
"papers excluded) were each filtered to abstracts containing "
"sensitization/nociplastic-pain-relevant keywords (e.g. \"central "
"sensitization,\" \"nociplastic,\" \"neuroinflammation,\" \"chronic "
"overlapping pain\"). Both filtered sets were embedded into one shared "
"semantic space (a single TF-IDF-plus-SVD model fit jointly on the union of "
"both sets, since two independently-fit embedding spaces are not comparable "
"by cosine similarity even when their dimensions coincide). The "
"highest-similarity cross-literature pairs were retained as candidates, "
"pairs already citing each other (checked via the free Semantic Scholar API) "
"were excluded as already-known connections rather than latent ones, and the "
"surviving candidates were sent to the same LLM for a final verification step: "
"does this specific pair of abstracts describe the same underlying mechanism, "
"conservatively judged, and if so, what is it. Two further checks were added "
"after early runs surfaced artifacts: a literal disease-name filter on both "
"single-disease corpora (mirroring 2.1's audit, since a paper's own single-topic "
"corpus is not automatically free of the same term-mapping contamination), "
"and an explicit same-PMID exclusion, after an early run's highest-scoring "
"\"connections\" turned out to be papers appearing, unlabeled, in both "
"corpora simultaneously, an artifact of PubMed's term expansion rather than a "
"genuine cross-literature finding."
)

h1("3. Results")

h2("3.1 Corpus scale and the false-positive audit")
p(
"The false-positive audit measured PubMed term-mapping contamination "
"directly rather than assuming it: 6 of 97 joint-corpus records (6.2%), "
"1,045 of 35,777 endometriosis-only records (2.9%), and 269 of 15,486 "
"fibromyalgia-only records (1.7%) did not literally contain their required "
"disease term(s) in the fetched title/abstract text, and were excluded. This "
"yielded final corpora of 91 joint-comorbidity papers, 34,732 "
"endometriosis-only papers, and 15,217 fibromyalgia-only papers. The higher "
"contamination rate in the joint (AND) query than in either single-term "
"query is itself informative: a two-term query has more independent "
"opportunities for term-mapping drift than a single-term query does. This "
"91-versus-tens-of-thousands asymmetry is a direct, measured confirmation of "
"the original mini-review's central claim: the endometriosis-fibromyalgia "
"comorbidity is a quantifiably thin sliver of two otherwise large literatures "
"(Figure 1)."
)

figure("fig1_corpus_audit.png",
       "Figure 1. Corpus scale for each of the three PubMed queries, before and "
       "after the literal-disease-term audit. The joint (comorbidity) corpus is "
       "two to three orders of magnitude smaller than either single-disease "
       "corpus; the joint query also shows the highest false-positive rate "
       "(6.2%), consistent with a two-term AND query having more independent "
       "opportunities for PubMed's automatic term-mapping to drift from literal "
       "text than a single-term query.")

h2("3.2 How much of the joint literature is actually about the comorbidity")
p(
"Structured extraction across all 91 joint-corpus papers shows that the "
"91-paper count itself overstates how much direct research attention the "
"comorbidity has received. Only 16 papers (17.6%) treat the "
"endometriosis-fibromyalgia comorbidity as their primary focus; 31 (34.1%) "
"discuss it as a secondary point; and the remaining 44 (48.4%), nearly half, "
"mention both conditions only incidentally, most often as two items in a "
"longer list of chronic overlapping pain conditions. Of the 91 papers, 48 "
"(52.7%) name no specific mechanism linking the two conditions at all; among "
"papers that do, central sensitization dominates (26 papers, 28.6%), followed "
"at much lower frequency by autoimmune (4), other named mechanisms (4), "
"nociplastic pain specifically (3), hormonal (3), neuropathic/small-fiber "
"pathology (2), and psychiatric comorbidity (1). On stance, 67 papers (73.6%) "
"do not address a shared-mechanism hypothesis at all in their own results or "
"discussion, versus 24 (26.4%) whose own findings support one; no paper in "
"the corpus explicitly argues against a shared mechanism. Study designs are "
"dominated by cross-sectional studies (24), narrative reviews (19), and "
"cohort studies (18), with comparatively few systematic reviews (6) and only "
"one randomized controlled trial protocol in the entire corpus (Figure 2)."
)

figure("fig2_extraction_breakdown.png",
       "Figure 2. Structured extraction results across all 91 joint-corpus "
       "papers. (a) How central the comorbidity is to each paper's own stated "
       "subject, most co-mentions are incidental. (b) The mechanism, if any, "
       "each paper proposes as linking the two conditions; over half name none.",
       width_in=6.3)

h2("3.3 Extraction validation")
p(
"Against 15 independently human-annotated papers, the final extraction "
"pipeline achieved almost-perfect agreement on mechanism classification "
"(93.3% raw agreement, Cohen's kappa = 0.89) and substantial agreement on "
"study design (80.0%, kappa = 0.75) and comorbidity focus (80.0%, kappa = "
"0.69). Sample size extraction matched exactly on all 8 papers where both the "
"model and the human annotator recorded a number (100%). Stance showed "
"moderate agreement (73.3-80.0% across iterations, kappa 0.44-0.58), with "
"disagreements concentrated on the genuinely ambiguous \"mixed/uncertain\" "
"category, a category human annotators themselves found difficult to apply "
"consistently during manual annotation. The fibromyalgia-measurement-type "
"field (distinguishing a genuine diagnosed comorbidity from a survey "
"instrument used only to quantify pain severity) was the weakest field "
"throughout: initial agreement was poor (46.7%, kappa = 0.20); tightening the "
"category definitions in the extraction prompt raised raw agreement to 60% "
"but did not improve chance-corrected agreement (kappa = 0.12); adding three "
"concrete worked examples of the exact disagreement pattern then raised "
"agreement to 80% with a real, if still modest, kappa of 0.32. This field is "
"reported here as an honest, partially-resolved limitation rather than a "
"solved problem: distinguishing a brief but genuine comorbidity mention from "
"a purely theoretical background reference evidently requires more context "
"than a short abstract reliably provides, even to a careful reader, and "
"further improvement likely requires access to full-text methods sections "
"rather than abstract-only extraction (Figure 3)."
)

figure("fig4_validation.png",
       "Figure 3. Extraction reliability per field, measured as Cohen's kappa "
       "against 15 independently human-annotated papers, with the standard "
       "Landis and Koch interpretation bands shown for reference. Mechanism "
       "and study design both reach substantial-to-almost-perfect agreement; "
       "the fibromyalgia-measurement-type field, reported here after two rounds "
       "of prompt revision, remains the weakest and is disclosed as such rather "
       "than omitted.",
       width_in=6.0)

h2("3.4 Topic structure of the joint literature")
p(
"BERTopic identified ten coherent topics across the 91-paper joint corpus, "
"including chronic overlapping pain conditions generally (14 papers), "
"bladder pain syndrome/interstitial cystitis (13), pelvic pain syndromes "
"(11), migraine comorbidity (8), a central-sensitization-specific cluster "
"explicitly co-locating sensitization terminology with endometriosis (7), "
"vulvodynia (6), autoimmune/systemic lupus erythematosus comorbidity (5), and "
"one topic whose top terms are \"women,\" \"fibromyalgia,\" \"endometriosis,\" "
"and \"prevalence\" directly, the clearest topic-level signature of papers "
"treating the comorbidity as such (5 papers). The classical LDA baseline, run "
"for comparison on the same corpus, produced markedly less separated topics "
"dominated by generic high-frequency terms (\"pain,\" \"chronic,\" \"women\"), "
"supporting embedding-based topic modeling as the more informative method for "
"a corpus of this size, consistent with the expectation that LDA requires "
"more documents than are available here to find stable topics (Figure 4)."
)

figure("fig3_topic_model.png",
       "Figure 4. BERTopic topic sizes for the 91-paper joint corpus, labeled "
       "by each topic's top four terms. One topic (\"women, fibromyalgia, "
       "endometriosis, women endometriosis\") is a direct topic-level signature "
       "of papers centered on the comorbidity itself.",
       width_in=6.3)

h2("3.5 Latent cross-literature connections")
p(
"After keyword-filtering for sensitization/nociplastic-pain-relevant language "
"and excluding joint-corpus papers, 2,549 fibromyalgia-only and 256 "
"endometriosis-only papers remained. Embedding both sets into one shared "
"semantic space and retaining the 30 highest-similarity cross-literature "
"pairs, all 30 survived the citation-overlap filter (none of the "
"highest-similarity pairs already cited each other), and the verification "
"step confirmed 22 of the 30 (73.3%) as describing a genuinely shared "
"mechanism. These 22 confirmed pairs are, by construction, papers that never "
"cite each other and never co-mention both conditions, yet independently "
"describe the same underlying biology. The single most-repeated mechanism "
"among the confirmed pairs is central sensitization and its associated "
"constructs (nociplastic pain, myofascial trigger points, chronic overlapping "
"pain conditions), appearing across roughly two-thirds of the confirmed "
"pairs. Two mechanism clusters recur independently across multiple "
"confirmed pairs rather than appearing once: the NLRP3 inflammasome/pyroptosis "
"pathway, connecting a fibromyalgia paper on AMPK-mediated NLRP3 activation "
"and a separate fibromyalgia paper on inflammasome biomarkers to the same "
"endometriosis paper on NLRP3-mediated pyroptosis; and gut-brain axis / "
"microbiome-mediated sensitization, connecting a fibromyalgia "
"gut-microbiome paper to an endometriosis paper on microbiome involvement in "
"pain pathogenesis. A further pair links growth hormone/IGF-1 signaling "
"literature from the two fields (Figure 5). Table 1 lists the ten "
"highest-confidence confirmed connections."
)

figure("fig5_discovery_network.png",
       "Figure 5. All 22 verified cross-literature connections, one "
       "fibromyalgia-literature paper linked to one endometriosis-literature "
       "paper per edge. No paper pair in this figure cites the other, and no "
       "paper here appears in the joint (co-mention) corpus; every connection "
       "shown is latent, recovered only through shared semantic content and "
       "confirmed by model-based mechanism verification. Edge thickness scales "
       "with verification confidence; color groups connections by shared "
       "mechanism.",
       width_in=6.0)

doc.add_paragraph()
tbl_cap = doc.add_paragraph()
tbl_run = tbl_cap.add_run(
    "Table 1. The ten highest-confidence verified connections between the "
    "fibromyalgia-only and endometriosis-only literatures. Confidence is the "
    "verifying model's self-reported confidence that the two abstracts "
    "describe the same mechanism."
)
tbl_run.italic = True
tbl_run.font.size = Pt(9.5)
table = doc.add_table(rows=1, cols=4)
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
hdr[0].text = "Shared mechanism"
hdr[1].text = "Fibromyalgia-literature paper"
hdr[2].text = "Endometriosis-literature paper"
hdr[3].text = "Confidence"

rows_data = [
("Central sensitivity syndromes / CSI",
 "The Central Sensitization Inventory (CSI): establishing clinically significant values for identifying central sensitivity syndromes in an outpatient chronic pain sample.",
 "Central sensitization inventory in endometriosis.",
 "1.00"),
("Nociplastic pain / central sensitization",
 "From fibrositis to fibromyalgia to nociplastic pain: how rheumatology helped get us here and where do we go from here?",
 "Nociplastic Pain in Endometriosis: A Scoping Review.",
 "0.99"),
("Myofascial trigger points and central sensitization",
 "Myofascial trigger points in migraine and tension-type headache.",
 "Relating Chronic Pelvic Pain and Endometriosis to Signs of Sensitization and Myofascial Pain and Dysfunction.",
 "0.95"),
("Myofascial trigger points and central sensitization",
 "Myofascial pain syndromes and their evaluation.",
 "Relating Chronic Pelvic Pain and Endometriosis to Signs of Sensitization and Myofascial Pain and Dysfunction.",
 "0.95"),
("Myofascial trigger points and central sensitization",
 "Trigger points--ultrasound and thermal findings.",
 "Relating Chronic Pelvic Pain and Endometriosis to Signs of Sensitization and Myofascial Pain and Dysfunction.",
 "0.95"),
("Myofascial trigger points and central sensitization",
 "Fibromyalgia, myofascial pain, tender points and trigger points: splitting or lumping?",
 "Relating Chronic Pelvic Pain and Endometriosis to Signs of Sensitization and Myofascial Pain and Dysfunction.",
 "0.95"),
("Nociplastic pain and central sensitization",
 "Nociplastic Pain: Facts, Controversies and Future Tasks.",
 "Nociplastic Pain in Endometriosis: A Scoping Review.",
 "0.95"),
("Central sensitization and somatic hyperalgesia",
 "Migraine and Central Sensitization: Clinical Features, Main Comorbidities and Therapeutic Perspectives.",
 "Impact of visceral pain on migraine symptoms in comorbid patients: a retrospective observational study.",
 "0.95"),
("NLRP3 inflammasome activation / IL-1b signaling",
 "AMPK Phosphorylation Modulates Pain by Activation of NLRP3 Inflammasome.",
 "Liriodendrin alleviates endometriosis via PPARgamma activation and inhibition of NLRP3 inflammasome-mediated pyroptosis.",
 "0.95"),
("Central sensitization",
 "Clinical and functional impact of central sensitization on patients with familial Mediterranean fever: a cross-sectional study.",
 "Prevalence and Risk Factors of Central Sensitization in Women with Endometriosis.",
 "0.95"),
]

for mech, fm, endo, conf in rows_data:
    row = table.add_row().cells
    row[0].text = mech
    row[1].text = fm
    row[2].text = endo
    row[3].text = conf

for row in table.rows:
    for cell in row.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)

doc.add_paragraph()

h1("4. Limitations")
p(
"Several limitations qualify these results. First, abstract-only extraction "
"necessarily misses information available only in full text: one "
"cross-literature discovery candidate (a mouse model of endometriosis pain) "
"turned out, on full-text inspection, to include a genuine human validation "
"cohort (19 endometriosis patients, 15 controls) that its abstract never "
"mentions, meaning any abstract-only pipeline, including this one, "
"systematically undercounts the true extent of human data in the literature. "
"Second, the fibromyalgia-measurement-type field remains only partially "
"resolved (kappa = 0.32 after two rounds of prompt revision), and should be "
"treated as a documented weak point rather than a validated result. Third, "
"the retrieval-augmented discovery step is explicitly conservative by "
"construction: it surfaces only high-similarity pairs among a "
"keyword-prefiltered subset of two large literatures, and is better "
"understood as a proof that such latent connections exist and are "
"findable than as an exhaustive map of every such connection. Fourth, this "
"analysis is restricted to published, indexed PubMed abstracts; it "
"necessarily inherits any publication bias, language bias, and indexing bias "
"already present in that literature, and says nothing about connections that "
"may exist only in preprints, grey literature, or unpublished data. Finally, "
"repeated re-extraction runs showed that two already-validated fields "
"(study design and stance) fluctuated by several percentage points between "
"runs using identical prompts, a reminder that LLM-based extraction, even "
"with structured output and a fixed prompt, is not perfectly deterministic, "
"and that any single validation run should be read as an estimate rather "
"than an exact figure."
)

h1("5. Conclusion")
p(
"The gap the original mini-review identified, no computational method has "
"treated the endometriosis-fibromyalgia comorbidity itself as the unit of "
"analysis, is real and measurable: fewer than one in five of the 91 papers "
"that even literally co-mention both conditions treat the comorbidity as "
"their actual subject. Closing that gap does not require new infrastructure. "
"A validated LLM extraction pipeline, a comparative topic-modeling analysis, "
"and a retrieval-augmented discovery method built entirely from free, "
"publicly available tools and public literature data were sufficient to "
"produce a quantified evidence base and, more importantly, 22 verified, "
"previously implicit connections between two literatures that have not yet "
"cited each other. The central sensitization framework, and specifically the "
"NLRP3 inflammasome pathway and gut-microbiome-mediated sensitization, stand "
"out as the most consistently recurring latent bridges, and are natural "
"starting points for the interdisciplinary, mechanism-driven research the "
"original mini-review called for."
)

doc.save(r"D:\Desktop\endo-fibro-lit-mining\article_draft_with_figures.docx")
print("saved")
