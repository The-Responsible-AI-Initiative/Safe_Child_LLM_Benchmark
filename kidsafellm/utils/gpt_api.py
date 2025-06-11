import openai
from pydantic import BaseModel
from typing import List

# 定义 Pydantic 模型
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

# OpenAI API 调用类
class OpenAIModel:
    def __init__(self, model_name, max_tokens=None, system_prompt='You are a helpful assistant.', temperature=0.0, top_p=0.9, **kwargs):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p

    def _compose_messages(self, content, image=None):
        if image is None:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ]
        else:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": content},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                ]}
            ]

    def _call_api_create(self, content):
        client = openai.OpenAI()
        return client.chat.completions.create(
            model=self.model_name,
            messages=self._compose_messages(content),
            max_tokens=self.max_tokens,
            temperature=self.temperature
        ).choices[0].message.content

    def _call_api_beta(self, content, response_format, beta_model="gpt-4o-2024-11-20", image=None):
        client = openai.OpenAI()
        return client.beta.chat.completions.parse(
            model=beta_model,
            messages=self._compose_messages(content, image=image),
            response_format=response_format
        ).choices[0].message.content

    def prompt(self, x):
        return self._call_api_create(x)

    def evolve_QA_Json(self, x):
        return self._call_api_beta(x, Json_QA_Harder)

    def annotate_w_Trans_Json(self, x):
        return self._call_api_beta(x, Json_Annotation_w_Translation)

    def annotate_explanation(self, x):
        return self._call_api_beta(x, Json_Annotation_Explanation)

    def evolve_prompt_I(self, x, image):
        return self._call_api_beta(x, Json_QA_Harder, image=image)

    def annotate_w_Trans_Json_I(self, x, image):
        return self._call_api_beta(x, Json_Annotation_w_Translation, image=image)

    def scenario(self, x):
        return self._call_api_beta(x, Json_Scenario, beta_model="gpt-4o-2024-11-20")

    def decision(self, x):
        return self._call_api_beta(x, Json_Decision, beta_model="gpt-4o-2024-11-20")

def get_embedding(texts):
    # 无需设置api_key，直接使用环境变量
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [response.data[i].embedding for i in range(len(texts))] 