import os
import json
import copy
import random
import re
import numpy as np

import gym
from gym import spaces
from stable_baselines3 import PPO
from together import Together

API_KEY = ""
client = Together(api_key=API_KEY)
MAX_EPISODE_STEPS = 50


def infer_field_types_from_gold(gold_labels):
    schema = {}
    sample = gold_labels[0]
    date_re = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}")
    for field, val in sample.items():
        if isinstance(val, bool):
            schema[field] = "bool"
        elif isinstance(val, str) and date_re.search(val):
            schema[field] = "date"
        else:
            schema[field] = "str"
    return schema


def generate_initial_schema(samples, task_description, gold_labels=None):
    prompt = (
        "You are an expert in schema generation. Based on the following sample data and task, "
        "please generate a JSON object representing the schema. The schema should map field names "
        "to their data types ('str', 'bool', 'date').\n\n"
        f"Samples: {samples}\n"
        f"Task: {task_description}\n\n"
        "Output only a JSON object."
    )
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": "You are an expert in schema extraction."},
                {"role": "user", "content": prompt}
            ],
        )
        text = response.choices[0].message.content
        matches = re.findall(r"\{[\s\S]*?\}", text)
        for block in matches:
            try:
                schema = json.loads(block)
                return schema
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print("LLM schema generation failed:", e)

    # Fallback to gold-driven schema if available
    if gold_labels:
        print("Falling back to schema inferred from gold labels.")
        return infer_field_types_from_gold(gold_labels)

    # As a last resort, minimal default
    print("Falling back to minimal default schema.")
    return {"field1": "str", "field2": "date", "field3": "bool"}


def infer_type_heuristic(sample_values, field_name=""):
    date_re = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}")
    truthy = {"true", "false", "yes", "no", "0", "1"}
    name_lower = field_name.lower()
    for val in sample_values:
        v = str(val).strip().lower()
        if date_re.search(v):
            return "date"
        if v in truthy or any(k in name_lower for k in ["flag", "is_", "has_"]):
            return "bool"
    return "str"


def extract_data_using_schema(schema, raw_data):
    records = []
    date_pattern = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
    for text in raw_data:
        rec = {}
        lower = text.lower()
        for field, ftype in schema.items():
            if ftype == "str":
                rec[field] = text[:100]
            elif ftype == "date":
                m = date_pattern.search(text)
                rec[field] = m.group(0) if m else ""
            elif ftype == "bool":
                rec[field] = field.lower() in lower
            else:
                rec[field] = None
        records.append(rec)
    return records


def evaluate_schema(schema, raw_data, gold_labels, initial_schema):
    """
    Simulated score: penalize missing or incorrect fields/types vs gold.
    """
    ideal = len(gold_labels[0]) if gold_labels else len(initial_schema)
    count_penalty = abs(len(schema) - ideal) * 0.05
    missing = [f for f in initial_schema if f not in schema]
    missing_penalty = len(missing) * 0.3
    type_penalty = sum(0.2 for f in schema if f in initial_schema and schema[f] != initial_schema[f])
    score = 1.0 - count_penalty - missing_penalty - type_penalty + random.uniform(-0.01, 0.01)
    score = max(0.0, min(1.0, score))
    print(f"Evaluated schema {schema} --> Simulated Score: {score:.3f}")
    return score


class SchemaRefinementEnv(gym.Env):
    """Gym env for schema refinement using gold labels for reward."""
    def __init__(self, init_schema, raw_data, gold_labels, task_description):
        super().__init__()
        self.initial_schema = copy.deepcopy(init_schema)
        self.schema = copy.deepcopy(init_schema)
        self.raw_data = raw_data
        self.gold_labels = gold_labels
        self.task_description = task_description
        self.candidate_pool = list(gold_labels[0].keys())
        self.critical_fields = set(self.candidate_pool)
        self.step_counter = 0
        self.base_score = evaluate_schema(self.schema, raw_data, gold_labels, self.initial_schema)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

    def _get_observation(self):
        return np.random.rand(10).astype(np.float32)

    def reset(self):
        self.schema = copy.deepcopy(self.initial_schema)
        self.step_counter = 0
        self.base_score = evaluate_schema(self.schema, self.raw_data, self.gold_labels, self.initial_schema)
        return self._get_observation()

    def step(self, action):
        self.step_counter += 1
        old_schema = copy.deepcopy(self.schema)
        reward = 0.0
        if action == 0:
            f = random.choice(self.candidate_pool)
            if f not in self.schema:
                self.schema[f] = "str"
                print(f"Action: Adding field '{f}'.")
            else:
                reward -= 0.2
        elif action == 1:
            if self.schema:
                f = random.choice(list(self.schema.keys()))
                if f in self.critical_fields:
                    reward -= 0.8
                else:
                    del self.schema[f]
                    print(f"Action: Removing field '{f}'.")
            else:
                reward -= 0.2
        elif action == 2:
            if self.schema:
                f = random.choice(list(self.schema.keys()))
                samples = [d.get(f, "") for d in self.gold_labels if f in d][:3]
                new_type = infer_type_heuristic(samples, f)
                old_type = self.schema[f]
                self.schema[f] = new_type
                print(f"Action: Modifying field '{f}' from {old_type} to {new_type}.")
            else:
                reward -= 0.2
        new_score = evaluate_schema(self.schema, self.raw_data, self.gold_labels, self.initial_schema)
        reward += new_score - self.base_score
        self.base_score = new_score
        done = self.step_counter >= MAX_EPISODE_STEPS
        return self._get_observation(), reward, done, {"old_schema": old_schema, "new_score": new_score}


def train_rl_agent(init_schema, raw_data, gold_labels, task_description, total_timesteps=1000):
    env = SchemaRefinementEnv(init_schema, raw_data, gold_labels, task_description)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return env.schema


def self_evolving_schema_pipeline(raw_data, samples, task_description, gold_labels):
    schema = generate_initial_schema(samples, task_description, gold_labels)
    print("Initial schema:", schema)
    init_score = evaluate_schema(schema, raw_data, gold_labels, schema)
    print("Initial schema evaluation score:", init_score)
    refined = train_rl_agent(schema, raw_data, gold_labels, task_description)
    final_score = evaluate_schema(refined, raw_data, gold_labels, schema)
    print("Final schema evaluation score:", final_score)
    return refined


if __name__ == "__main__":
    raw_data = [
        "From: john.doe@example.com\nTo: fraudteam@example.com\nSubject: Suspected Fraud Activity\n\nHi team, ...",
        "From: news@corporatenews.com\nTo: all@example.com\nSubject: Company Update\n\nAccording to a report...",
        "From: jane.smith@example.com\nTo: fraudteam@example.com\nSubject: Follow-up on Expense Anomalies\n\nDear Team, ..."
    ]
    gold_labels = [
        {"sender": "john.doe@example.com", "recipient": "fraudteam@example.com", "subject": "Suspected Fraud Activity", "body": "Hi team, ...", "mentions_fraud": True, "quotes_news": False, "date": "2024-01-01"},
        {"sender": "news@corporatenews.com", "recipient": "all@example.com", "subject": "Company Update", "body": "According to a report...", "mentions_fraud": False, "quotes_news": True, "date": "2024-01-02"},
        {"sender": "jane.smith@example.com", "recipient": "fraudteam@example.com", "subject": "Follow-up on Expense Anomalies", "body": "Dear Team, ...", "mentions_fraud": True, "quotes_news": False, "date": "2024-01-03"}
    ]
    samples = raw_data[:2]
    task_description = "Extract key fields from emails including sender, recipient, subject, body and date."
    # task_description = "Extract key information from structured documents"
    final_schema = self_evolving_schema_pipeline(raw_data, samples, task_description, gold_labels)
    print("\nOptimized Schema After Self-Evolution:")
    print(json.dumps(final_schema, indent=2))
