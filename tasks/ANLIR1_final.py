#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

##############################################################################################################################

# ################
# import os
# import json
# model_name_question = os.environ['EXP_MODE_QUESTION']
# # question_file = '/nvmedata/changranh/ama_question_synthetic_data/' + model_name_question + self.task_name + ".jsonl"
# question_file = '/scratch/changranh/ama_question_synthetic_data/' + model_name_question + "_ANLIR1" + ".jsonl"        
# ################  


##############################################################################################################################
# All prompts
questioner_prompt = InputOutputPrompt(
    input_formatter=lambda x: f"Statement: {x['statement']}",
    output_formatter=lambda x: f"Question: {x['question']}",
    required_keys=["question", "statement"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Rewrite the statement as a yes/no question.\n\n"
)
questioner_prompt_examples = [
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
        {
            "statement": "She ate the entire cake.",
            "question": "Did she eat the entire cake?",
        },
        {
            "statement": "The cat drank all of the milk.",
            "question": "Did the cat drink all of the milk?",
        },
        {
            "statement": "The dog barked at the postman.",
            "question": "Did the dog bark at the postman?",
        },
        {
            "statement": "John walked to the store.",
            "question": "Did John walk to the store?",
        },
        {
            "statement": "The children laughed at the clown.",
            "question": "Did the children laugh at the clown?"
        }
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
        {
            "statement": "The teacher gave a quiz today.",
            "question": "Did the teacher give a quiz today?",
        },
        {
            "statement": "The basketball team won the game.",
            "question": "Did the basketball team win the game?",
        },
        {
            "statement": "The cat was sleeping in the corner.",
            "question": "Was the cat sleeping in the corner?",
        },
        {
            "statement": "The ocean is a beautiful place to visit.",
            "question": "Is the ocean a beautiful place to visit?",
        },
        {
            "statement": "The girl painted a picture of a house.",
            "question": "Did the girl paint a picture of a house?"
        }
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
        {
            "statement": "I love going to the movies",
            "question": "Do you love going to the movies?",
        },
        {
            "statement": "She wants to go to school.",
            "question": "Does she want to go to school?",
        },
        {
            "statement": "There are 8 planets in the solar system.",
            "question": "Are there 8 planets in the solar system?",
        },
        {
            "statement": "I have a red car",
            "question": "Do you have a red car?",
        },
        {
            "statement": "My cat has four legs",
            "question": "Does your cat have four legs?"
        }
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
        {
            "statement": "The snow was very deep.",
            "question": "How deep was the snow?",
        },
        {
            "statement": "He has been to France twice.",
            "question": "How many times has he been to France?",
        },
        {
            "statement": "She was the only one who got the answer right.",
            "question": "Who was the only one who got the answer right?",
        },
        {
            "statement": "She has been living in the city for 5 years.",
            "question": "How long has she been living in the city?"
        },
        {
            "statement": "The ice cream was very sweet.",
            "question": "How sweet was the ice cream?"
        }
    ]),
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test not hard?",
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?",
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?",
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?",
        },
        {
            "statement": "She likes to play the piano.",
            "question": "Does she like to play the piano?"
        },
        {
            "statement": "He drives a car.",
            "question": "Does he drive a car?"
        },
        {
            "statement": "I am going to the store.",
            "question": "Are you going to the store?"
        },
        {
            "statement": "We have been friends for a long time.",
            "question": "Have you been friends for a long time?"
        },
        {
            "statement": "The cat is sleeping.",
            "question": "Is the cat sleeping?"
        }
    ]),
]

extraction_qa = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['context']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["context", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer the question. If there is no evidence in the context, return \"Unknown\".\n\n"
)
extraction_qa_examples = [
    pd.DataFrame([
        {
            "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
            "question": "Based on the context, Did the plague affect people in Europe?",
            "answer": "yes, people in Italy, Europe",
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown",
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "Based on the context, Is anti-matter made of electrons? ",
            "answer": "Unknown",
        },
        {
            "context": "The Industrial Revolution was a period of dramatic change, marked by the introduction of new manufacturing processes and the development of new sources of energy.",
            "question": "Based on the context, Was the Industrial Revolution a period of technological advancement?",
            "answer": "Yes",
        },
        {
            "context": "The Internet is a global network of computers that use the Internet Protocol Suite, a set of communications protocols, to link billions of devices worldwide.",
            "question": "Based on the context, Was the Internet created to connect computers?",
            "answer": "Yes",
        },
        {
            "context": "In the early 20th century, Albert Einstein proposed the theory of relativity, which states that the laws of physics remain the same in all inertial frames of reference.",
            "question": "Based on the context, Did Albert Einstein propose the theory of gravity?",
            "answer": "No",
        },
        {
            "context": "The English language is a West Germanic language that is spoken by more than 400 million people in the world.",
            "question": "Based on the context, Is English a Germanic language?",
            "answer": "Yes",
        },
        {
            "context": "The Big Bang Theory is a cosmological model of the observable universe from the earliest known periods through its subsequent large-scale evolution.",
            "question": "Based on the context, Does the Big Bang Theory explain the formation of the universe?",
            "answer": "Yes",
        },
        {
            "context": "The greenhouse effect is the process by which radiation from the atmosphere warms a planet's surface to a temperature above what it would be without its atmosphere.",
            "question": "Based on the context, Does the greenhouse effect increase a planet's temperature?",
            "answer": "Yes",
        },
        {
            "context": "The Roman Empire was one of the largest empires in ancient history and spanned over three continents.",
            "question": "Based on the context, Did the Roman Empire extend over more than one continent?",
            "answer": "Yes"
        }
    ]),
    pd.DataFrame([
        {
            "context": "According to Biraben, the plague was present somewhere in Italy only between 1346 and 1671, and not after that.",
            "question": "Based on the context, Was the plague present in Italy during the 2000s?",
            "answer": "No, it was present between 1346 and 1671"
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "Based on the context, Is anti-matter made of electrons? ",
            "answer": "Unknown"
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown"
        },
        {
            "context": "The brain is composed of a large number of highly specialized cells called neurons.",
            "question": "Based on the context, How many types of cells are in the brain?",
            "answer": "Many types, specifically neurons"
        },
        {
            "context": "The human heart is a four-chambered organ made up of cardiac muscle.",
            "question": "Based on the context, What type of organ is the human heart?",
            "answer": "A four-chambered organ made up of cardiac muscle"
        },
        {
            "context": "The United Nations is an intergovernmental organization that seeks to promote international cooperation and peace.",
            "question": "Based on the context, What does the United Nations do?",
            "answer": "It seeks to promote international cooperation and peace"
        },
        {
            "context": "The Earth's atmosphere is divided into five main layers: troposphere, stratosphere, mesosphere, thermosphere, and exosphere.",
            "question": "Based on the context, What is the highest layer of the Earth's atmosphere?",
            "answer": "The exosphere"
        },
        {
            "context": "Chlorine is a chemical element with atomic number 17 and symbol Cl.",
            "question": "Based on the context, What is the atomic number of chlorine?",
            "answer": "17"
        },
        {
            "context": "The element carbon exists in nature in several forms, including graphite, diamond, and amorphous carbon.",
            "question": "Based on the context, What elements can carbon be found in?",
            "answer": "Graphite, diamond, and amorphous carbon"
        },
        {
            "context": "The United States Constitution was written in 1787 and ratified in 1788.",
            "question": "Based on the context, When was the United States Constitution ratified?",
            "answer": "1788"
        }
    ]),
    pd.DataFrame([
        {
            "context": "Jenna's 10th birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "Based on the context, Did 10 friends attend Jenna's party?",
            "answer": "Unknown"
        },
        {
            "context": "The bullies attacked John when he was walking through the elementary school parking lot and then got sent to the teacher's office.",
            "question": "Based on the context, Did the bullies attack John in the teacher's office?",
            "answer": "No, parking lot"
        },
        {
            "context": "WISS discovered a new monkey disease occurring in a remote tribe in the Amazon rainforrest.",
            "question": "Based on the context, Did WISS discover a new monkey species?",
            "answer": "No, a new monkey disease"
        },
        {
            "context": "The researchers used a new chemical compound to treat the patients suffering from cancer.",
            "question": "Based on the context, Did the researchers use a chemical compound to cure cancer?",
            "answer": "No, to treat"
        },
        {
            "context": "The police found a suspicious package in the abandoned warehouse.",
            "question": "Based on the context, Did the police find a dangerous package in the warehouse?",
            "answer": "Unknown"
        },
        {
            "context": "The little girl was very scared when the loud thunderstorm started.",
            "question": "Based on the context, Did the little girl get scared by the thunder?",
            "answer": "Yes"
        },
        {
            "context": "The actor was nominated for an Oscar for his performance in the movie.",
            "question": "Based on the context, Did the actor receive an Oscar for the movie?",
            "answer": "Unknown"
        },
        {
            "context": "The firemen managed to save five people from the burning building.",
            "question": "Based on the context, Did the firemen save three people from the building?",
            "answer": "No, five people"
        },
        {
            "context": "The archaeologist found an ancient temple buried under the sand.",
            "question": "Based on the context, Did the archaeologist find a pyramid buried under the sand?",
            "answer": "No, a temple"
        },
        {
            "context": "The hikers were warned about the dangerous animals that lived in the area.",
            "question": "Based on the context, Were the hikers warned about the wild plants in the area?",
            "answer": "No, dangerous animals"
        }
    ]),
    pd.DataFrame([
        {
            "context": "When Judy and Jack went to school, they got in trouble with their teacher for being late. I didn't think it was very fair.",
            "question": "Based on the context, Did she think it was fair?",
            "answer": "No"
        },
        {
            "context": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
            "question": "Based on the context, Is inflation good for society?",
            "answer": "Unknown"
        },
        {
            "context": "Put yourself out there. The more time you spend dating and socializing, the more likely you will find a boyfriend you like.",
            "question": "Based on the context, Does socializing help you find a boyfriend?",
            "answer": "Yes"
        },
        {
            "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
            "question": "Based on the context, Did the plague affect people in Europe?",
            "answer": "yes, people in Italy, Europe",
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown",
        },
        {
            "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
            "question": "Based on the context, Is anti-matter made of electrons? ",
            "answer": "Unknown",
        },
        {
            "context": "The current global population is estimated to be over 7 billion people.",
            "question": "Based on the context, Is the global population growing?",
            "answer": "Yes"
        },
        {
            "context": "A study last year showed that the number of people using public transportation in the city had decreased by 3%.",
            "question": "Based on the context, Has the use of public transportation decreased?",
            "answer": "Yes"
        },
        {
            "context": "Yoga is a form of exercise that helps to increase flexibility and strength.",
            "question": "Based on the context, Does yoga help with flexibility?",
            "answer": "Yes"
        },
        {
            "context": "An apple a day keeps the doctor away.",
            "question": "Based on the context, Does an apple a day help with health?",
            "answer": "Yes"
        }
    ]),
    pd.DataFrame([
         {
            "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
            "question": "Based on the context, Did the plague affect over 1,000 people?",
            "answer": "yes, 1,200 people",
        },
        {
            "context": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
            "question": "Based on the context, Is inflation good for society?",
            "answer": "Unknown"
        },
        {
            "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
            "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
            "answer": "unknown"
        },
        {
            "context": "Da Vinci was a great painter, inventor and scientist from the Renaissance period.",
            "question": "Based on the context, Was Da Vinci a famous artist?",
            "answer": "Yes"
        },
        {
            "context": "The World Health Organization has reported that the Covid-19 pandemic has caused over 1 million deaths around the world.",
            "question": "Based on the context, Has the Covid-19 pandemic caused over 1 million deaths?",
            "answer": "Yes, over 1 million deaths"
        },
        {
            "context": "The United States is a federal republic composed of 50 states, a federal district, five major self-governing territories, and various possessions",
            "question": "Based on the context, Does the United States consist of more than 50 states?",
            "answer": "Yes, the United States consists of 50 states, a federal district, five major self-governing territories, and various possessions."
        },
        {
            "context": "It has been found that people who exercise regularly have improved mental health and are less likely to suffer from depression.",
            "question": "Based on the context, Does exercise have a positive effect on mental health?",
            "answer": "Yes, exercise has been found to have a positive effect on mental health and reduce the risk of depression."
        },
        {
            "context": "The human body is composed of 60% water. ",
            "question": "Based on the context, Is the human body mostly composed of water?",
            "answer": "Yes, the human body is composed of 60% water."
        },
        {
            "context": "The world's coral reefs are in danger due to climate change, pollution, and unsustainable fishing practices.",
            "question": "Based on the context, Are coral reefs threatened by human activities?",
            "answer": "Yes, coral reefs are threatened by climate change, pollution, and unsustainable fishing practices."
        },
        {
            "context": "The government is introducing a new education policy to increase the quality of teaching in schools.",
            "question": "Based on the context, Is the government trying to improve the quality of teaching?",
            "answer": "Yes, the government is introducing a new education policy to increase the quality of teaching in schools."
        }
    ]),
]

 

class ANLIR1Decomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = [' True', ' False', ' Neither']
        num_per_class = int(np.ceil(k_shot / len(labels)))
        print(f"Selecting {num_per_class} examples per class.")

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            sub_df = train_data[train_data["targets_pretokenized"] == label].sample(
                num_per_class
            )
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break
        mini_df = pd.concat(dfs)
        return mini_df

    def _get_boost_decomp_examples(self, train_data, boost_id):
        seed = [69, 987][boost_id] 
        k_shot = 64
        random.seed(seed)
        np.random.seed(seed)

        data_train = pd.DataFrame(train_data)
        labels = [' Neither', ' False', ' True']
        num_per_class = int(np.ceil(k_shot / len(labels)))

        dfs = []
        total_in_context = 0
        for label in labels:
            while num_per_class + total_in_context > k_shot:
                num_per_class -= 1
            if seed % 2 == 1:
                sub_df = data_train[data_train["targets_pretokenized"] == label].sample(num_per_class, random_state = seed)
            elif seed % 2 == 0:
                sub_df = data_train[data_train["targets_pretokenized"] != label].sample(num_per_class, random_state = seed)
            dfs.append(sub_df)
            total_in_context += num_per_class
            if total_in_context == k_shot:
                break

        booster_df = pd.concat(dfs).sample(frac=1, random_state=0)
        print(f"Selected: {len(booster_df)} in context examples.")
        return [
            booster_df
        ]

    def get_boost_decomp_examples(self, train_data, boost_id):
        if boost_id < 3:
            return [
                questioner_prompt_examples[boost_id],
                extraction_qa_examples[boost_id],
            ]
        else:
            icl_examples = self._get_boost_decomp_examples(train_data, boost_id-3)[0]
            return [
                icl_examples
            ]

    def zero_few_baseline(
        self,
        test_data,
        few_shot_df,
        manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer,
        do_few_shot=True,
    ):
        expt_log = {}
        golds = []
        preds = []

        labels = set(test_data["targets_pretokenized"])
        labels = [l.lower().strip() for l in labels]

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if ind in expt_log:
                pred = expt_log[ind]["pred"]
                gold = expt_log[ind]["gold"]
            else:
                icl_str = ""

                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        if len(tokenizer.encode(icl_str, truncation=False)) >= 3500:
                            break                        
                        icl_str += f"{s_row['inputs_pretokenized']}{s_row['targets_pretokenized']}\n\n"

                text = row["inputs_pretokenized"]
                text = text.replace("True, False, or Neither?", "").strip().strip("\n")
                text = text + " True, False, or Neither? "
                gold = row["targets_pretokenized"]
                prompt = f"{icl_str}{{text:}}"
                pmp = prompt.format(text=text)
                if i == 0:
                    print(pmp)

                raw_answer = get_response(
                    pmp,
                    manifest_answer,
                    overwrite=bool(overwrite_manifest_answer),
                    max_toks=20,
                )
                answer = raw_answer.strip().lower()
                answer = answer.split("\n")
                answer = [a for a in answer if a]
                answer = [
                    a for a in answer if any(l.lower() in a.lower() for l in labels)
                ]
                if answer:
                    answer = answer[0]
                else:
                    answer = ""
                answer = "".join(
                    [a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']]
                )
                is_yes = "true" in answer.split()
                is_no = "false" in answer.split()
                is_maybe = "neither" in answer.split()
                pred = "Neither"
                if is_yes and (not is_maybe and not is_no):
                    pred = "True"
                if is_no and (not is_maybe and not is_yes):
                    pred = "False"
                if is_maybe and (not is_no and not is_yes):
                    pred = "Neither"

                gold = gold.strip().lower()
                pred = pred.strip().lower()
                entry = {
                    "ind": ind,
                    "example": text,
                    "base_prompt": pmp,
                    "raw_answer": raw_answer,
                    "pred": pred,
                    "gold": gold,
                }
                expt_log[ind] = entry

            golds.append(gold)
            preds.append(pred)

        report = classification_report(golds, preds, output_dict=True)
        return expt_log, report["accuracy"]

    def get_extraction(self, question, passage, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        if "Based on the context," in prompt_suffix:
            question_prefix = " Based on the context,"
        else:
            question_prefix = ""
        extract_prompt = f"{prompt_suffix}\n\nContext: {{passage:}}\nQuestion:{question_prefix} {question}\nAnswer:"
        extract_pmp = extract_prompt.format(passage=passage)
        answer = get_response(
            extract_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        answer = answer.replace(",", "").replace(".", "").replace("?", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0]
        else:
            answer = passage
        return answer, extract_pmp

    def get_question(self, statement, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        question_prompt = f"{prompt_suffix}\n\nStatement: {{statement:}}\nQuestion:"
        question_pmp = question_prompt.format(statement=statement)
        answer = get_response(
            question_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
           
        
        answer = answer.replace("Question: ", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0].strip()
        # answer = ''
        statement = statement.strip().strip(".")
        if (
            not answer
            or statement.lower() == answer.lower()
            or not answer.strip().endswith("?")
        ):
            answer = f"{statement}. Yes, no, or unknown?"
        answer = answer.split("\n")[0]
        
        # ####################
        # with open(question_file, 'a') as f:
        #     json_string = json.dumps({'prompt': question_pmp, "completion":answer})
        #     f.write(json_string + '\n')             
        # ####################   
        
        return answer, question_pmp

    def resolve_pred(self, answer):
        is_yes = "yes" in answer.split() or "true" in answer.split()
        is_no = "no" in answer.split() or "false" in answer.split()
        is_maybe = "maybe" in answer.split() or "maybe" in answer.split()

        pred = "Neither"
        if is_yes and (not is_maybe and not is_no):
            pred = "True"
        if is_no and (not is_maybe and not is_yes):
            pred = "False"
        if is_maybe and (not is_no and not is_yes):
            pred = "Neither"

        return pred

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=-1)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=1000)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train, indecisive_ans="neither")
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            report = classification_report(labels, [p[i] for p in all_boost_preds], output_dict=True)
            individual_accuracies.append(report["accuracy"])
            print(report)
            print("\n\n")
        report = classification_report(labels, preds, output_dict=True)
        print(report)
        return expt_log, expt_log_train, report["accuracy"], individual_accuracies    
    
    def _run_decomp_single_data(
        self, test_data, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit = -1
    ):
        expt_log = {}
        all_boost_preds = []
        labels = []       
        
        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            prompts_across_boost = []
            preds_across_boost = []

            if i == run_limit:
                break
            
            text = row["inputs_pretokenized"]
            gold = row["targets_pretokenized"].strip()
            passage = text.split("\n")[0]
            statement = (
                text.split("\n")[-1]
                .replace("True, False, or Neither?", "")
                .strip()
                .strip("\n")
                .replace("Question: ", "")
            )
            for boost_num, boost_examples in enumerate(boost_dfs):
                all_prompts = []

                # question / extract prompt
                if boost_num < 3:
                    question, question_final_prompt = self.get_question(
                        statement, questioner_prompt, boost_examples[0], manifest_question, overwrite_manifest_question
                    )
                     
                    
                    all_prompts.append(question_final_prompt)

                    open_answer_f, extraction_final_prompt = self.get_extraction(
                        question,
                        passage,
                        extraction_qa,
                        boost_examples[1],
                        manifest_answer,
                        overwrite_manifest_answer,
                    )
                    all_prompts.append(extraction_final_prompt)
                    if i == 0:
                        print("\n".join(all_prompts))
                    answer_f = open_answer_f.lower()
                    pred = self.resolve_pred(answer_f)
                    pred = pred.strip().lower()

                    preds_across_boost.append(pred)

                # just ICL
                else:
                    icl_str = ""
                    for s_ind, s_row in boost_examples[0].iterrows():
                        if s_row["targets_pretokenized"].strip() == "True":
                            demo_label = "yes"
                        elif s_row["targets_pretokenized"].strip()  == "False":
                            demo_label = "no"
                        else:
                            demo_label = "unknown"

                        s_text = s_row["inputs_pretokenized"]
                        s_passage = s_text.split("\n")[0]
                        s_statement = (
                            s_text.split("\n")[-1]
                            .replace("True, False, or Neither?", "")
                            .strip()
                            .strip("\n")
                            .replace("Question: ", "")
                        )
                        icl = f"Statement: {s_statement}\nAnswer: {demo_label}"
                        icl_str += f"{icl}\n\n"

                    description = "Is the statement Yes, No, or Unknown?"
                    prompt = f"{description}\n\n{icl_str}Statement: {{statement:}}\nAnswer:"
                    pmp = prompt.format(statement=statement)
                    if i == 0:
                        print("PMP ICL")
                        print(pmp)
                    pred = get_response(
                        pmp,
                        manifest_answer,
                        overwrite=bool(overwrite_manifest_answer),
                        max_toks=10,
                        stop_token="\n",
                    )
                    pred = pred.lower().strip()
                    pred = pred.replace(".", "").replace(",", "").replace("Label: ", "").replace("Sentiment:", "")
                    pred = [p for p in pred.split("\n") if p]
                    if pred:
                        pred = pred[0]
                    else:
                        pred = ""

                    all_prompts.append(pmp)
                    prompts_across_boost.append(all_prompts)
                    pred = self.resolve_pred(pred).lower()
                    preds_across_boost.append(pred)
                gold = gold.strip().lower()

            expt_log[ind] = {
                "ind": ind,
                "preds_boost": preds_across_boost,
                "prompts": prompts_across_boost,
                "example": text,
                "pred": pred,
                "gold": gold,
            }
            all_boost_preds.append(preds_across_boost)
            labels.append(gold)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 5
    task_name = "anli_r1"
    data_dir = f"{DATA_DIR}/P3/data_feather/anli_GPT_3_style_r1"
    decomp = ANLIR1Decomp(task_name, data_dir, val_split="test")
    decomp.run(args)


if __name__ == "__main__":
    main()