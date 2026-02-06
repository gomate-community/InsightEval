Title: 

**InsightEval: An Automated System for Evaluating Insightfulness in Scientific Papers**

Abstract：

> Assessing the insightfulness of scientific writing is critical for understanding whether a paper offers new understanding beyond summarizing prior work. However, insight is inherently relative to existing literature and is rarely captured by evaluations that focus on writing quality or contribution overlap, leaving insight assessment largely underexplored.
We present InsightEval, an automated system for evaluating the insightfulness of scientific papers by assessing how much new understanding they provide beyond their cited references. Our approach is inspired by a cognitive hypothesis: after thoroughly reading all references cited by a paper, the extent to which reading the paper introduces new understanding reflects its level of insight.
The system operates through four stages: (1) extracting opinion sentences from the target paper; (2) retrieving supporting materials from cited references for each opinion sentence via semantic retrieval; (3) scoring each opinion sentence across four insight dimensions—information gain, depth, breadth, and height—using a large language model conditioned on the retrieved supporting materials; and (4) synthesizing the sentence-level evaluations into a paper-level insight report.
We deploy InsightEval on over 100 human-written and 100 AI-generated scientific papers, demonstrating its applicability across diverse writing sources. For transparency and inspection, we publicly release the InsightEval source code, a demonstration video, and all generated insight evaluation reports.