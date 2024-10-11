from anthropic import AnthropicVertex
import bigframes.pandas as bpd
from datasets import load_dataset
from google.auth import default, transport
from IPython.display import HTML, Markdown, display
import openai
from vertexai.evaluation import (
    EvalTask,
    MetricPromptTemplateExamples,
    PointwiseMetric
)

from vertexai.generative_models import GenerativeModel

from model_mgmt import testing, config, instructions, prompt
import logging
import pandas as pd
import plotly.graph_objects as go
import random
import string
import warnings
import vertexai

# flake8: noqa --E501

project_number = testing.project_number
gemma_endpoint_id = testing.gemma_endpoint_id
gemini_tuned_endpoint_id = testing.gemini_tuned_endpoint_id
PROJECT_ID = testing.project_id
LOCATION = testing.location
COMPUTATIONAL_EXPERIMENT_NAME = "flash-1.5-002-eval"
POINTWISE_EXPERIMENT = "flash-1.5-002-pointwise-eval"

DEST_DATASET = "model_evaluation"


vertexai.init(project=PROJECT_ID, location=LOCATION)


logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def display_explanations(eval_result, metrics=None, n=1):
    """Display the explanations."""
    style = "white-space: pre-wrap; width: 1500px; overflow-x: auto;"
    metrics_table = eval_result.metrics_table
    df = metrics_table.sample(n=n)

    if metrics:
        df = df.filter(
            ["response", "baseline_model_response"]
            + [
                metric
                for metric in df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )
    for index, row in df.iterrows():
        for col in df.columns:
            display(HTML(f"{col}:{row[col]}"))
        display(HTML(""))


def display_eval_result(eval_result, title=None, metrics=None):
    """Display the evaluation results."""
    summary_metrics, metrics_table = (
        eval_result.summary_metrics,
        eval_result.metrics_table,
    )

    metrics_df = pd.DataFrame.from_dict(summary_metrics, orient="index").T
    if metrics:
        metrics_df = metrics_df.filter(
            [
                metric
                for metric in metrics_df.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )
        metrics_table = metrics_table.filter(
            [
                metric
                for metric in metrics_table.columns
                if any(selected_metric in metric for selected_metric in metrics)
            ]
        )

    if title:
        # Display the title with Markdown for emphasis
        display(Markdown(f"## {title}"))
    # Display the summary metrics DataFrame
    display(Markdown("### Summary Metrics"))
    display(metrics_df)
    # Display the metrics table DataFrame
    display(Markdown("### Row-based Metrics"))
    display(metrics_table)


def display_radar_plot(eval_results, metrics=None):
    """Plot the radar plot."""
    fig = go.Figure()
    for item in eval_results:
        title, eval_result = item
        summary_metrics = eval_result.summary_metrics
        if metrics:
            summary_metrics = {
                k.replace("/mean", ""): summary_metrics[k]
                for k, v in summary_metrics.items()
                if any(selected_metric + "/mean" in k for selected_metric in metrics)
            }
        fig.add_trace(
            go.Scatterpolar(
                r=list(summary_metrics.values()),
                theta=list(summary_metrics.keys()),
                fill="toself",
                name=title,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])), showlegend=True
    )
    fig.show()


def display_bar_plot(eval_results_list, metrics=None):
    """Plot the bar plot."""
    fig = go.Figure()
    data = []

    for eval_results in eval_results_list:
        title, eval_result = eval_results[0], eval_results[1]

        summary_metrics = eval_result.summary_metrics
        mean_summary_metrics = [f"{metric}/mean" for metric in metrics]
        updated_summary_metrics = []
        if metrics:
            for k, v in summary_metrics.items():
                if k in mean_summary_metrics:
                    updated_summary_metrics.append((k, v))
            summary_metrics = dict(updated_summary_metrics)
            # summary_metrics = {k: summary_metrics[k] for k, v in summary_metrics.items() if any(selected_metric in k for selected_metric in metrics)}

        data.append(
            go.Bar(
                x=list(summary_metrics.keys()),
                y=list(summary_metrics.values()),
                name=title,
            )
        )

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(barmode="group", showlegend=True)
    fig.show()


def generate_uuid(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


ds = (
    load_dataset(
        "Open-Orca/OpenOrca",
        data_files="1M-GPT4-Augmented.parquet",
        split="train[:100]",
    )
    .to_pandas()
    .drop(columns=["id"])
    .rename(columns={"response": "reference"})
)

bpd.options.bigquery.project = PROJECT_ID
bpd.options.bigquery.location = LOCATION

tools = config.model_to_call(config.Selected_Model)[2]

model = GenerativeModel(
    "gemini-1.5-flash-002",
    generation_config=config.generation_config,
    safety_settings=config.safety_settings,
    tools=tools
)


def comp_eval():
    comp_query = f'''
    SELECT
    prompt, label as reference, """{prompt.template}""" AS context, """{instructions.system_instructions}""" AS instructions
    FROM
    `{PROJECT_ID}.model_fine_tuning.eval_cleaned`
    '''

    bq_df = bpd.read_gbq(comp_query)

    # eval_dataset = eval_dataset_gbq.to_pandas

    # # dataset = ds.sample(n=10)
    eval_dataset = pd.DataFrame({
        "reference": bq_df["reference"],
        "instructions": bq_df["instructions"],
        "context": bq_df["context"],
        "prompt": bq_df["prompt"],
    })

    rouge_eval_task = EvalTask(
        dataset=eval_dataset,
        metrics=["rouge_l_sum"],
        # experiment=COMPUTATIONAL_EXPERIMENT_NAME,
    )
    rouge_result = rouge_eval_task.evaluate(
        model=model,
        prompt_template="{instructions} {context}"
    )

    rouge_result_output = bpd.DataFrame(rouge_result.metrics_table)

    rouge_result_output.to_gbq(f"{PROJECT_ID}.{DEST_DATASET}.compute_results_latest",
                               if_exists="replace")


comp_eval()


def pointwise_eval():
    query = f'''
    SELECT
    message_text as prompt, reply_text as reference, """{prompt.template}""" AS context, """{instructions.system_instructions}""" AS instructions
    FROM
    `{PROJECT_ID}.chat_app_lineage.messages` m
    JOIN
    `{PROJECT_ID}.chat_app_lineage.replies` r
    ON m.session_id = r.session_id AND m.message_count = r.reply_count
    '''

    pointwise_dataset_gbq = bpd.read_gbq(query)
    # pointwise_dataset = pointwise_dataset_gbq.to_pandas

    pointwise_dataset = pd.DataFrame({
        "prompt": pointwise_dataset_gbq["prompt"],
        "reference": pointwise_dataset_gbq["reference"],
        "context": pointwise_dataset_gbq["context"],
        "instructions": pointwise_dataset_gbq["instructions"],
    })

    what_to_eval = [
        "groundedness",
        "instruction_following",
        "safety"
    ]

    metrics_dict = {}

    for item in what_to_eval:
        metrics_dict[item] = f"{item}_metric"

    output_metrics = []

    for key in metrics_dict:
        metric_name = metrics_dict[key]
        locals()["metric_name"] = PointwiseMetric(
            metric=key,
            metric_prompt_template=MetricPromptTemplateExamples.get_prompt_template(key)  # noqa -E501
        )
        output_metrics.append(key)

    print(output_metrics)

    pointwise_eval_task = EvalTask(
        dataset=pointwise_dataset,
        metrics=output_metrics,
        # experiment=POINTWISE_EXPERIMENT,
    )

    pointwise_result = pointwise_eval_task.evaluate(
        model=model,
        prompt_template="{instructions} {context} {prompt}"
    )

    pointwise_result_output = bpd.DataFrame(pointwise_result.metrics_table)

    pointwise_result_output.to_gbq(f"{PROJECT_ID}.{DEST_DATASET}.pointwise_results_latest",
                                   if_exists="replace")

    print(pointwise_result.metrics_table.head())


pointwise_eval()
