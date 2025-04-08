+++
title = "ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Format Restriction, and Column Exploration"
date = 2025-04-07T12:00:00-08:00
authors = ["Minghang Deng", "Ashwin Ramachandran", "Canwen Xu", "Lanxiang Hu", "Zhewei Yao", "Anupam Datta", "Hao Zhang"]
author = "Minghang Deng, Ashwin Ramachandran, Canwen Xu, Lanxiang Hu, Zhewei Yao, Anupam Datta, Hao Zhang"
ShowReadingTime = true
draft = false
[cover]
      image = "img/cover_reforce.png"
      alt = "cover_reforce"
      caption = "An instance of the ReFoRCE workflow operates as follows: upon receiving a question, it generates the expected answer format, systematically explores relevant tables and columns, engages in self-refinement, and ultimately decides on a final answer once self-consistency is achieved."
+++

{{< socialBadges arxiv-index="2502.00675" github="hao-ai-lab/ReFoRCE" >}}

{{< justify >}}

**TL;DR:** Text-to-SQL systems have transformed data access by enabling natural language queries over structured databases, but their deployment in enterprise settings is hindered by challenges such as massive schemas (1,000+ columns), diverse SQL dialects (e.g., BigQuery, Snowflake), and complex analytical queries—reflected in the Spider 2.0 benchmark, where state-of-the-art performance remains limited (~25%) due to poor instruction-following, limited long-context understanding, weak self-refinement, and lack of dialect-specific knowledge. Addressing these limitations, we propose ReFoRCE (Self-Refinement Agent with Format Restriction and Column Exploration), which introduces table compression to mitigate long-context issues, format restriction to ensure accurate SQL answer generation, and iterative column exploration for enhanced schema understanding. Complementing these features, a self-refinement pipeline with parallelized workflows using voting mechanisms further resolves challenging cases, ultimately achieving state-of-the-art scores of 31.26 on Spider 2.0-Snow and 30.35 on Spider 2.0-Lite.
{{< /justify >}}

## Background: Toy-Level vs. Enterprise-Grade Text-to-SQL Tasks

{{< justify >}}
Text-to-SQL systems have made querying structured data more accessible through natural language, but transitioning from toy-level datasets to enterprise-grade applications reveals major challenges. Early benchmarks like [Spider 1.0](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/) provided valuable starting points, but they fall short in complexity—featuring relatively small schemas (e.g., ~27 and ~54 columns per DB) and limited SQL functionality (0.0–0.4 functions per query), with minimal support for external knowledge or diverse SQL dialects. In contrast, **Spider 2.0** and its variants (Spider 2.0-lite and Spider 2.0-snow) introduce realistic enterprise scenarios, with databases averaging over **800 columns**, significantly longer SQL queries (up to **161.8 tokens per query**), richer function use, and support for multiple SQL dialects, external knowledge, and project-level complexity. Despite their importance, performance on Spider 2.0 remains capped at ~25%, underscoring the need for systems that can handle ambiguous queries, long contexts, and dialect-specific reasoning in real-world environments.
{{< /justify >}}

### Conventional Text-to-SQL Benchmarks

#### Spider 1.0

{{< justify >}}
Spider 1.0 has long served as a benchmark for Text-to-SQL tasks. It consists of non-industrial databases that typically feature only a few tables and columns, with SQL queries that are relatively straightforward. Early research in this area predominantly focused on model training and fine-tuning strategies, aiming to convert natural language questions into SQL queries.

**One example of Spider 1.0:** Which country is the driver with the highest points from? Give me the capital of the country.
{{< /justify >}}

```SQL
SELECT T1.Capital 
FROM country AS T1 
JOIN driver AS T2 ON T1.Country_ID  =  T2.Country 
ORDER BY T2.Points DESC LIMIT 1;
```

#### BIRD

{{< justify >}}
Alongside Spider 1.0, BIRD has emerged as a widely recognized benchmark. Recent advances in prompting with large language models (LLMs) have enabled methods to achieve impressive performance on BIRD, reaching approximately 70% accuracy. This benchmark brings to light new challenges, including noisy database values, the need for external knowledge, and SQL efficiency. However, similar to Spider 1.0, BIRD is constructed on relatively simplistic databases, which limits its ability to capture the full complexity of real-world scenarios.

**One example of BIRD:** How many cards did Volkan BaÇµa illustrated whose foreign language is in French?
{{< /justify >}}

```SQL
SELECT COUNT(DISTINCT c."id") AS "Number_of_Cards"
FROM "cards" c
JOIN "foreign_data" f ON c."uuid" = f."uuid"
WHERE c."artist" = 'Volkan Baǵa' AND f."language" = 'French';
```

### Spider 2.0

{{< justify >}}
The advent of powerful LLMs has shifted Text-to-SQL research from fine-tuning to prompting, yielding impressive results on simplified datasets like Spider 1.0 (over 90% accuracy). However, these datasets—with limited tables, columns, and simple SQL—don’t reflect real-world complexities. [Spider 2.0](https://spider2-sql.github.io/) addresses these gaps by incorporating multiple SQL dialects, nested columns, external knowledge, and ambiguous queries, challenging models to handle more realistic scenarios.

**One example of Spider 2.0:** Calculate the net difference between the number of pancreatic adenocarcinoma (PAAD) patients in TCGA's dataset who are confirmed to have mutations in both KRAS and TP53 genes, and those without mutations in either gene. Utilize patient clinical and follow-up data alongside genomic mutation details from TCGA’s cancer genomics database, focusing specifically on PAAD studies where the mutations have passed quality filters.
{{< /justify >}}

```SQL
WITH paad_patients AS (
    SELECT DISTINCT "bcr_patient_barcode" AS "ParticipantBarcode"
    FROM PANCANCER_ATLAS_2.PANCANCER_ATLAS.FILTERED_CLINICAL_PANCAN_PATIENT_WITH_FOLLOWUP
    WHERE "acronym" = 'PAAD'
), 
kras_mutations AS (
    SELECT DISTINCT "ParticipantBarcode"
    FROM PANCANCER_ATLAS_2.PANCANCER_ATLAS.FILTERED_MC3_MAF_V5_ONE_PER_TUMOR_SAMPLE
    WHERE "Study" = 'PAAD' AND "FILTER" = 'PASS' AND "Hugo_Symbol" = 'KRAS'
),
tp53_mutations AS (
    SELECT DISTINCT "ParticipantBarcode"
    FROM PANCANCER_ATLAS_2.PANCANCER_ATLAS.FILTERED_MC3_MAF_V5_ONE_PER_TUMOR_SAMPLE
    WHERE "Study" = 'PAAD' AND "FILTER" = 'PASS' AND "Hugo_Symbol" = 'TP53'
),
both_mutations AS (
    SELECT "ParticipantBarcode"
    FROM kras_mutations
    INNER JOIN tp53_mutations USING ("ParticipantBarcode")
),
neither_mutations AS (
    SELECT paad."ParticipantBarcode"
    FROM paad_patients paad
    LEFT JOIN kras_mutations kras ON paad."ParticipantBarcode" = kras."ParticipantBarcode"
    LEFT JOIN tp53_mutations tp53 ON paad."ParticipantBarcode" = tp53."ParticipantBarcode"
    WHERE kras."ParticipantBarcode" IS NULL AND tp53."ParticipantBarcode" IS NULL
)
SELECT 
    (SELECT COUNT(*) FROM both_mutations) - (SELECT COUNT(*) FROM neither_mutations) AS "Net_difference"
```

### Limitations of Current Method
{{< justify >}}
Realistic Text-to-SQL challenges demand agentic methods that enable LLMs to dynamically interact with databases by leveraging tools, executing commands, and planning actions. Such approaches tackle tasks ranging from planning and reasoning to advanced code generation, as evidenced by frameworks like [Spider-Agent](https://arxiv.org/abs/2411.07763) built on the ReAct paradigm for Spider 2.0. However, code agents often struggle to maintain control in long-context scenarios, frequently requiring iterative corrections for syntax errors, data types, and function selection. Moreover, current Text-to-SQL methods face difficulties handling multiple dialects, nested columns, and complex data types in the Spider 2.0 dataset.
{{< /justify >}}

{{< image src="img/example_spider_agent.png" alt="spider_agent" width="120%" title="Figure 1: A base case for Spider Agent: improperly handling nested columns without self-refinement.">}}

## ReFoRCE

### Overview

{{< justify >}}
ReFoRCE offers a robust solution to the unpredictability and lack of reliability in ReAct agents by introducing a structured and controlled workflow. By breaking down tasks into manageable subtasks, employing self-refinement, and utilizing format restriction and column exploration, ReFoRCE significantly enhances the agent's ability to identify and address challenging examples. The integration of parallelization and a voting mechanism further improves the reliability of the outcomes. ReFoRCE’s flexibility and ease of integration with various database systems make it a versatile and scalable solution. This methodology provides a high level of consistency and reliability, even when faced with difficult datasets.
{{< /justify >}}

{{< image src="img/cover_reforce.png" alt="cover_reforce" width="130%" title="Figure 2: An overview of the Self-Refinement Agent with Format Restriction and Column Exploration (ReFoRCE) workflow. (a) Table compression to address long-context limitations; (b) Format restriction to ensure accurate answer formatting; (c) Iterative column exploration for improved schema understanding; (d) Self-refinement pipeline comprising parallelized workflows with voting mechanisms.">}}

### Table Information Compression

{{< justify >}}
Following the approach of Spider 2.0, we create a dictionary for each example using Database Definition Language (DDL) files that include external knowledge and table structures. When DDL files exceed setting size, we apply pattern-based matching to merge tables with similar prefixes or suffixes, retaining only one representative DDL file. For others, we provide only table names. For example, the `GA360` database includes tables from `GA_SESSIONS_20160801` to `GA_SESSIONS_20170801`, each with DDL files over 150 KB, totaling more than 50 MB. Our pattern-based compression significantly reduces the size of such databases. Also a simple API-based schema linking ensures the context length remains manageable for long prompts. Given the database information $\mathcal{D}$ and auxiliary documentation $\mathcal{E}$, we apply the $\texttt{compress}$ function and concatenate the result with the question $\mathcal{Q}$ to form the initial input prompt $\mathcal{P}_{\text{init}}$: 
{{< /justify >}}

$$
\begin{align}
\mathcal{P}_{\text{init}} = \texttt{compress}(\mathcal{D}) + \mathcal{E} + \mathcal{Q}.
\end{align}
$$

### Expected Answer Format Restriction

{{< justify >}}
Realistic Text-to-SQL problems often face challenges with long contexts exceeding 100k tokens, leading to loss of critical information and inaccurate outputs. To address this, we propose Expected Answer Format Restriction, which involves generating and reinforcing the expected answer format (e.g., column names, data types, rows) at the outset and during self-refinement. The response must follow a strict CSV format, explicitly defining columns and ensuring each record is on a separate row. Specific cases like superlatives, percentages, and coordinates are handled clearly, and ambiguous terms may prompt additional columns for precision. For an LLM chat session $\mathcal{L}_{\text{session}}$, we input initial and format prompts $\mathcal{P}\_{\text{format}}$ to generate the expected answer format $\mathcal{F}$:  
{{< /justify >}}

$$
\begin{align}
\mathcal{F} = \mathcal{L}_{\text{session}}((\mathcal{P}\_{\text{init}}, \mathcal{P}\_{\text{format}})).
\end{align}
$$

### Exploration of Potentially Useful Columns

{{< justify >}}
When directly providing database information to an LLM, lack of details on value types and SQL dialects often leads to syntax errors and incorrect function calls, which are time-consuming to correct. To address these issues, we design an approach to explore relevant columns, beginning with simple SQL queries that gradually increase in complexity. For example, in Snowflake dialect cases, the queries follow the structure `SELECT DISTINCT "COLUMN_NAME" FROM DATABASE.SCHEMA.TABLE WHERE ...` and use techniques like `LATERAL FLATTEN` for nested columns and fuzzy string matching for flexibility. We also use an additional LLM chat session $\mathcal{L}'\_{\text{session}}$ with column exploration prompts $\mathcal{P}\_{\text{exploration}}$, to generate relevant tables, columns $\mathcal{P}\_{\text{column}}$, and SQL queries $\mathcal{S}\_{\text{exploration}}$. The resulting queries are executed using database APIs to retrieve results $\mathcal{R}\_{\text{exploration}}$:
{{< /justify >}}
$$
\begin{align}
\mathcal{P}\_{\text{column}}, \mathcal{S}\_{\text{exploration}} = \mathcal{L}'\_{\text{session}}(\mathcal{P}\_{\text{init}}, \mathcal{P}\_{\text{exploration}}),
\mathcal{R}\_{\text{exploration}} = \texttt{API}(\mathcal{S}\_{\text{exploration}}).
\end{align}
$$

### Self-Refinement Workflow for Problem-Solving

{{< justify >}}
After obtaining the table information $\mathcal{P}\_{\text{init}}$, exploring value data $\mathcal{P}\_{\text{column}} + \mathcal{R}\_{\text{exploration}}$, and defining the expected answer format $\mathcal{F}$, we input these elements into the model and apply a self-refinement process to correct errors and achieve high-confidence results $\mathcal{R}\_{\text{final}}$ through self-consistency:
{{< /justify >}}
$$
\begin{align}
\mathcal{R}\_{\text{final}} = \texttt{self-refinement}(\mathcal{P}\_{\text{init}}, \mathcal{P}\_{\text{column}} + \mathcal{R}\_{\text{exploration}}, \mathcal{F}).
\end{align}
$$

### Parallelization

{{< justify >}}
Despite the self-consistency mechanism, variations in results may occur due to different perspectives in column exploration. To improve confidence, we parallelize the workflow by running multiple threads simultaneously and use a voting mechanism to identify the most likely correct outcome. If discrepancies remain, the model further evaluates the results. Given the same format $\mathcal{F}$ and database information $\mathcal{P}\_{\text{init}}$, we launch multiple threads (set to 3 in our experiments) to execute the process above independently in parallel. The results from parallel execution are aggregated through a voting mechanism:
{{< /justify >}}
$$
\begin{align}
\mathcal{R}\_{\text{vote}} = \texttt{model_vote}(\mathcal{R}\_{\text{final1}}, \mathcal{R}\_{\text{final2}}, \mathcal{R}\_{\text{final3}}).
\end{align}
$$
{{< justify >}}
Additionally, since each example is independent and both the model and database APIs support parallel execution, we enable parallelization across examples, accelerating the process while ensuring consistency.
{{< /justify >}}

##  Experiments

### Results
{{< justify >}}
We evaluate our approach using the Spider 2.0 dataset, which includes two subsets: Spider 2.0-Snow and Spider 2.0-Lite, each with 547 examples and over 150 databases. The key difference between the subsets lies in their SQL dialects: Spider 2.0-Snow focuses on Snowflake, while Spider 2.0-Lite supports BigQuery, Snowflake, and SQLite. Using Execution Accuracy (EX) as the metric, our method achieves EX scores of 31.26 on Spider 2.0-Snow and 30.35 on Spider 2.0-Lite with the $\texttt{o1-preview}$ model, outperforming all other methods. 
{{< /justify >}}

### 🧊 Spider 2.0-Snow Leaderboard
{{< center >}}
| Rank | Date       | Method                                              | Score |
|------|------------|-----------------------------------------------------|--------|
| 1    | Jan 28, 2025 | **ReFoRCE** + o1-preview | **31.26** |
| 2    | Mar 8, 2025 | Spider-Agent + Claude-3.7-Sonnet-20250219          | 24.50  |
| 3    | Mar 16, 2025 | Spider-Agent + Claude-3.7-Sonnet-20250219-Thinking | 24.31  |
| 4    | Nov 30, 2024 | Spider-Agent + o1-preview                          | 23.58  |
| 5    | Feb 11, 2025 | Spider-Agent + o1-2024-12-17                       | 23.21  |
| 6    | Feb 1, 2025  | Spider-Agent + o3-mini-2025-01-31                 | 19.20  |
| 7    | Mar 7, 2025  | Spider-Agent + Claude-3.5-Sonnet-20241022 | 19.01  |
| 8    | Feb 10, 2025 | Spider-Agent + Claude-3.5-Sonnet-20241022         | 15.54  |
| 9    | Mar 13, 2025 | Spider-Agent + Gemini-2.0-Pro                     | 13.89  |
| 10   | Nov 30, 2024 | Spider-Agent + GPT-4o-2024-11-20                  | 12.98  |
{{< /center >}}

### 💡 Spider 2.0-Lite Leaderboard
{{< center >}}
| Rank | Date       | Method                                              | Score |
|------|------------|-----------------------------------------------------|--------|
| 1    | Jan 28, 2025 | **ReFoRCE** + o1-preview | **30.35** |
| 2    | Mar 16, 2025 | Spider-Agent + Claude-3.7-Sonnet-20250219-Thinking | 28.52  |
| 3    | Mar 8, 2025  | Spider-Agent + Claude-3.7-Sonnet-20250219         | 25.41  |
| 4    | Mar 28, 2025 | LinkAlign + DeepSeek-V3   | 24.86  |
| 5    | Feb 10, 2025 | Spider-Agent + o3-mini-2025-01-31                 | 23.40  |
| 6    | Nov 30, 2024 | Spider-Agent + o1-preview                          | 23.03  |
| 7    | Mar 10, 2025 | Spider-Agent + DeepSeek-R1                        | 13.71  |
| 8    | Feb 10, 2025 | Spider-Agent + GPT-4o-2024-11-20                  | 13.16  |
| 9    | Mar 13, 2025 | Spider-Agent + QwQ-32B                            | 11.33  |
| 10   | Dec 31, 2024 | Duo                               | 8.96   |
{{< /center >}}

{{< justify >}}
*Spider 2.0-Snow and Spider 2.0-Lite Leaderboards, March 2025.*
{{< /justify >}}

### Case Study
{{< justify >}}
Here is an example where ReFoRCE succeeds. The question is the same as in Figure 1, where Spider Agent struggles with handling nested columns in the Snowflake dialect. ReFoRCE first produces the expected answer format, titled "System,Most_Frequent_License". Compared to Spider Agent's answer, "System,LICENSENAME,LICENSECOUNT", ReFoRCE's output aligns better with the question description. During the column exploration stage, ReFoRCE receives SQL execution feedback starting with simpler queries and progressing to more complex ones. Even if there are errors, it can self-correct to avoid the issues seen in Spider Agent's output. After gaining sufficient knowledge about the database, ReFoRCE generates the final answer and verifies the executed results through its self-refinement and self-consistency workflow.
{{< /justify >}}

{{< image src="img/example_reforce.png" alt="reforce" width="120%" title="Figure 3: A successful case for ReFoRCE.">}}

## Get started
{{< justify >}}
For more details, please see [our paper](https://arxiv.org/abs/2502.00675). We also invite you to explore [our codebase](https://github.com/hao-ai-lab/ReFoRCE)!
{{< /justify >}}

## Citation

```
@article{deng2025reforce,
  title={ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Format Restriction, and Column Exploration},
  author={Deng, Minghang and Ramachandran, Ashwin and Xu, Canwen and Hu, Lanxiang and Yao, Zhewei and Datta, Anupam and Zhang, Hao},
  journal={arXiv preprint arXiv:2502.00675},
  year={2025}
}
```
