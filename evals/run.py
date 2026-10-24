from langsmith import Client

from evals.dataset import CRM_AGENT_EVAL_DATASET
from evals.evaluators import companies_match, contacts_match, no_forbidden_tools, tools_match
from evals.target import run_agent

DATASET_NAME = "crm-agent"

if __name__ == "__main__":
    client = Client()

    # Upload only once, otherwise every run would duplicate the examples
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(DATASET_NAME, description="CRM agent scenarios")
        examples = [
            # LangSmith calls the expected values "outputs" and hands them to evaluators as `reference_outputs`
            {"inputs": s["inputs"], "outputs": s["reference_outputs"], "metadata": s["metadata"]}
            for s in CRM_AGENT_EVAL_DATASET
        ]
        client.create_examples(dataset_id=dataset.id, examples=examples)

    client.evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[companies_match, contacts_match, tools_match, no_forbidden_tools],
        experiment_prefix="baseline",
        metadata={"model": "claude-haiku-5-5", "prompt": "v1"},
        max_concurrency=2,
    )
