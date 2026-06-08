# InsightEval: An Automated System for Evaluating Insightfulness in Scientific Papers

## Abstract

Assessing the insightfulness of scientific writing is critical for understanding whether a paper offers new understanding beyond summarizing prior work. However, insight is inherently relative to existing literature and is rarely captured by evaluations that focus on writing quality or contribution overlap, leaving insight assessment largely underexplored.

We present InsightEval, an automated system for evaluating the insightfulness of scientific papers by assessing how much new understanding they provide beyond their cited references. Our approach is inspired by a cognitive hypothesis: after thoroughly reading all references cited by a paper, the extent to which reading the paper introduces new understanding reflects its level of insight.

The system operates through four stages: (1) extracting opinion sentences from the target paper; (2) retrieving supporting materials from cited references for each opinion sentence via semantic retrieval; (3) scoring each opinion sentence across four insight dimensions—information gain, depth, breadth, and height—using a large language model conditioned on the retrieved supporting materials; and (4) synthesizing the sentence-level evaluations into a paper-level insight report.

We deploy InsightEval on over 200 human-written and AI-generated survey papers, and conduct sampled human validation showing moderate-to-strong alignment with human judgments. To support transparency and reproducibility, we release the InsightEval studio, library, and demonstration video.

* Code: https://github.com/gomate-community/InsightEval
* Video: https://drive.google.com/file/d/1cSGcskSnL5V251q1t7peQZ9Fs5uB1YbA/view?usp=sharing

![demo.png](resources/demo.png)

## Running InsightEval Locally

### 1. Clone the Repository

```bash
git clone https://github.com/gomate-community/InsightEval
```

### 2. Start the Backend

Open a terminal and run:

```bash
cd InsightEval/backend
pip install -r requirements.txt
```

Then start the backend service according to the backend entry point provided in the repository.

### 3. Start the Frontend

Open a new terminal and run:

```bash
cd InsightEval/frontend
npm install
npm run dev
```

By default, the demo page can be accessed at:

```text
http://localhost:5174/demo
```


## System Pipeline

### Stage 1: Opinion Sentence Extraction

![step1.png](resources/step1.png)

### Stage 2: Evidence Retrieval via RAG

![step2.png](resources/step2.png)

### Stage 3: Multi-Dimensional Insight Scoring

![step3.png](resources/step3.png)

### Stage 4: Paper-Level Report Synthesis

![step4.png](resources/step4.png)

