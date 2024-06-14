import boto3
import json

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

def invoke(prompt):

    systemPrompt = """
                    You have the personality of a charming southern gentleman.
                   """;

    prompt_config = {
        "prompt": f'<s>[INST]{systemPrompt} {prompt}[/INST]',
        "max_tokens": 1024,
        "temperature": 0.8
    }

    response = bedrock_runtime.invoke_model(
        body=json.dumps(prompt_config),
        modelId="mistral.mistral-large-2402-v1:0"
    )

    response_body = json.loads(response.get("body").read())
    return response_body["outputs"][0]["text"]
