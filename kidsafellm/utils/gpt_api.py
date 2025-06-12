'''Environment variables required (configure them in kidsafellm/config.py or
export in your shell):
    OPENAI_API_KEY      – GPT-4o
    ANTHROPIC_API_KEY   – Claude 3 Sonnet
    GOOGLE_API_KEY      – Gemini Flash/Pro
    TOGETHER_API_KEY    – Llama-3-70B (Together.ai host)
    GROK_API_KEY        – xAI Grok-3
    DEEPINFRA_API_KEY   – DeepInfra host for Vicuna, Mistral, DeepSeek
'''

import os
import json
from typing import List, Dict

import openai                    # OpenAI & Azure OpenAI
import anthropic                 # Claude
import google.generativeai as genai  # Gemini
import requests                  # REST helpers for Together, DeepInfra, Grok
from pydantic import BaseModel

#Pydantic model
class Json_QA(BaseModel):
    Question: str
    Explanation: str
    Correct_Answer: str


class Json_Evol(BaseModel):
    Methods_List: str
    Plan: str
    Rewritten_Instruction: Json_QA
    Finally_Rewritten_Instruction: Json_QA


class Json_Annotation(BaseModel):
    Category: List[str]
    Topic: str
    Knowledge_Points: List[str]
    Number_of_Knowledge_Points_Needed: int


class Json_QA_Harder(BaseModel):
    Correct_Answer: str
    Assessment_of_Incorrect_Options_Difficulty: str
    Replacement_of_Easiest_to_Judge_Options_with_Relevant_Knowledge_Points: str
    Modified_Question: str
    Explanation: str


class Json_Annotation_w_Translation(BaseModel):
    Explanation: str
    Question_in_Chinese: str
    Explanation_in_Chinese: str
    Category: List[str]
    Topic: str
    Knowledge_Points: List[str]
    Number_of_Knowledge_Points_Needed: int


class Json_Annotation_Explanation(BaseModel):
    Explanation: str
    Explanation_in_Chinese: str


class Json_Decision(BaseModel):
    class DecisionDetail(BaseModel):
        Decision: str
        Consequence: str

    Decisions: List[DecisionDetail]


class Json_Scenario(BaseModel):
    class LabSafetyIssues(BaseModel):
        Most_Common_Hazards: List[str]
        Improper_Operation_Issues: List[str]
        Negative_Lab_Environment_Impacts: List[str]
        Most_Likely_Safety_Incidents: List[str]

    Scenario: str
    LabSafety_Related_Issues: LabSafetyIssues
    Categories: List[str]
    Topic: str
    SubCategory: str

# OpenAI API 
class OpenAIModel:
    def __init__(
        self,
        model_name: str,
        max_tokens: int | None = None,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.0,
        top_p: float = 0.9,
        **kwargs,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p


    # Internal helpers

    def _compose_messages(self, content: str, image: str | None = None):
        if image is None:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ]
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ],
            },
        ]

    def _call_api_create(self, content: str) -> str:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=self._compose_messages(content),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def _call_api_beta(
        self,
        content: str,
        response_format: type[BaseModel],
        beta_model: str = "gpt-4o-2024-11-20",
        image: str | None = None,
    ) -> str:
        client = openai.OpenAI()
        resp = client.beta.chat.completions.parse(
            model=beta_model,
            messages=self._compose_messages(content, image=image),
            response_format=response_format,
        )
        return resp.choices[0].message.content

    def prompt(self, x: str) -> str:
        return self._call_api_create(x)

    def evolve_QA_Json(self, x: str) -> str:
        return self._call_api_beta(x, Json_QA_Harder)

    def annotate_w_Trans_Json(self, x: str) -> str:
        return self._call_api_beta(x, Json_Annotation_w_Translation)

    def annotate_explanation(self, x: str) -> str:
        return self._call_api_beta(x, Json_Annotation_Explanation)

    def evolve_prompt_I(self, x: str, image: str) -> str:
        return self._call_api_beta(x, Json_QA_Harder, image=image)

    def annotate_w_Trans_Json_I(self, x: str, image: str) -> str:
        return self._call_api_beta(x, Json_Annotation_w_Translation, image=image)

    def scenario(self, x: str) -> str:
        return self._call_api_beta(x, Json_Scenario, beta_model="gpt-4o-2024-11-20")

    def decision(self, x: str) -> str:
        return self._call_api_beta(x, Json_Decision, beta_model="gpt-4o-2024-11-20")


# Embedding helper
def get_embedding(texts: List[str]) -> List[List[float]]:
    client = openai.OpenAI()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [resp.data[i].embedding for i in range(len(texts))]

'''benchmark wrappers setup for each proprietary async models and local processing models'''

# GPT-4o (OpenAI)
def call_gpt4o(prompt: str, cfg: Dict | None = None) -> str:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rsp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return rsp.choices[0].message.content


# Claude 3.7 Sonnet  (Anthropic)
def call_claude_sonnet(prompt: str, cfg: Dict | None = None) -> str:
    cl = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    rsp = cl.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return rsp.content[0].text


# Gemini 2.0 Flash-Pro 
def call_gemini_flash(prompt: str, cfg: Dict | None = None) -> str:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    rsp = model.generate_content(prompt, generation_config={"temperature": 0.2})
    return rsp.text


# Llama-3-70B (Together.ai) 
def call_llama_70b(prompt: str, cfg: Dict | None = None) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "meta-llama/Llama-3-70B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    url = "https://api.together.xyz/v1/chat/completions"
    rsp = requests.post(url, headers=headers, json=data, timeout=60)
    rsp.raise_for_status()
    return rsp.json()["choices"][0]["message"]["content"]


# Grok-3 (xAI)
def call_grok3(prompt: str, cfg: Dict | None = None) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROK_API_KEY')}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "grok-1.5",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }
    rsp = requests.post(
        "https://api.grok.x.ai/v1/chat/completions", headers=headers, json=body, timeout=60
    )
    rsp.raise_for_status()
    return rsp.json()["choices"][0]["message"]["content"]


# DeepInfra generic helper for Vicuna / Mistral / DeepSeek
def _deepinfra_call(repo: str, prompt: str) -> str:
    url = f"https://api.deepinfra.com/v1/inference/{repo}"
    headers = {"Authorization": f"Bearer {os.getenv('DEEPINFRA_API_KEY')}"}
    body = {
        "input": prompt,
        "max_new_tokens": 1024,
        "temperature": 0.2,
    }
    rsp = requests.post(url, headers=headers, json=body, timeout=60)
    rsp.raise_for_status()

    # DeepInfra returns {"results":[{"generated_text": "..."}], ...}
    return rsp.json()["results"][0]["generated_text"]


def call_vicuna_7b(prompt: str, cfg: Dict | None = None) -> str:
    return _deepinfra_call("lmsys/vicuna-7b-v1.5", prompt)


def call_mistral_7b(prompt: str, cfg: Dict | None = None) -> str:
    return _deepinfra_call("mistralai/Mistral-7B-Instruct-v0.3", prompt)


def call_deepseek_r1(prompt: str, cfg: Dict | None = None) -> str:
    return _deepinfra_call("deepseek-ai/DeepSeek-R1", prompt)

#exported dispatch table
MODEL_DISPATCH: Dict[str, callable] = {
    "gpt4o_response":             call_gpt4o,
    "llama3.1-70B_response":      call_llama_70b,
    "claude3.7sonnet_response":   call_claude_sonnet,
    "gemini2.0flashpro_response": call_gemini_flash,
    "grok3_response":             call_grok3,
    "vicuna-7B_response":         call_vicuna_7b,
    "mistral-7B_response":        call_mistral_7b,
    "deepseekr1_response":        call_deepseek_r1,
}

# convenience list for iteration ordering
MODEL_COLUMNS: List[str] = list(MODEL_DISPATCH.keys())