#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
import pandas as pd

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt


from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


questioner = InputOutputPrompt(
    input_formatter=lambda x: f"Statement: {x['statement']}",
    output_formatter=lambda x: f"Question: {x['question']}",
    required_keys=["statement", "question"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Rewrite the statement as a yes/no question.\n\n"
)

questioner_examples = [
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not",
            "question": "Was the test hard?"
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?"
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?"
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?"
        } ,
        {
            "statement": "The boy was happy",
            "question": "Was the boy happy?"
        },
        {
            "statement": "The teacher gave out the papers",
            "question": "Did the teacher give out the papers?"
        },
        {
            "statement": "The shower was cold",
            "question": "Was the shower cold?"
        },
        {
            "statement": "John ate the cake",
            "question": "Did John eat the cake?"
        },
        {
            "statement": "The family went to the beach",
            "question": "Did the family go to the beach?"
        }
    ]).sample(frac=1).reset_index(drop=True),     
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not",
            "question": "Was the test hard?"
        },
        {
            "statement": "it is a good idea to buy your parents gifts",
            "question": "Is it a good idea to buy your parents gifts?"
        },
        {
            "statement": "the balloon popped",
            "question": "Did the balloon pop?"
        },
        {
            "statement": "The father and son went camping to California.",
            "question": "Did the father and son go camping?"
        },
        {
            "statement": "Cats are cute and cuddly.",
            "question": "Are cats cute and cuddly?"
        },
        {
            "statement": "The sky is blue.",
            "question": "What color is the sky?"
        },
        {
            "statement": "The sun rises in the east.",
            "question": "Which direction does the sun rise?"
        },
        {
            "statement": "The dog can fetch a ball.",
            "question": "Can the dog fetch a ball?"
        },
        {
            "statement": "The cat slept on the bed.",
            "question": "Where did the cat sleep?"
        }
    ]).sample(frac=1).reset_index(drop=True),
    pd.DataFrame([
        {
            "statement": "she prefers kittens over puppies.",
            "question": "Does she prefer kittens over puppies?",
        },
        {
            "statement": "Max and his wife went on a trip to Europe",
            "question": "Did Max and his wife go on a trip to Europe?",
        },
        {
            "statement": "jared was born during the war in 1942.",
            "question": "Was Jared born during a war in 1942?",
        },
        {
            "statement": "it took jenna 7 attempts to solve the problem",
            "question": "Did it take Jenna 7 attempts to solve the problem?",
        },
        {
            "statement": "The team won the championship",
            "question": "Did the team win the championship?",
        },
        {
            "statement": "The bird was singing in the tree",
            "question": "Was the bird singing in the tree?",
        },
        {
            "statement": "Spencer has been living in London for 5 years",
            "question": "Has Spencer been living in London for 5 years?",
        },
        {
            "statement": "The show starts at 8 pm",
            "question": "Does the show start at 8 pm?",
        },
        {
            "statement": "The restaurant was crowded",
            "question": "Was the restaurant crowded?",
        },
        {
            "statement": "The sun is shining today",
            "question": "Is the sun shining today?",
        }
    ]).sample(frac=1).reset_index(drop=True),
]

openended_qa = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['passage']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["passage", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Provide the answer to the question from the passage.\n\n"
)

openended_qa_examples = [
    pd.DataFrame([
        {
            "passage": "When Judy and Jack went to school, they got in trouble with their teacher for being late. I didn't think it was very fair.",
            "question": "Did she think it was fair?",
            "answer": "No"
        },
        {
            "passage": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
            "question": "Is inflation good for society?",
            "answer": "Maybe"
        },
        {
            "passage": "Put yourself out there. The more time you spend dating and socializing, the more likely you will find a boyfriend you like.",
            "question": "Does socializing help you find a boyfriend?",
            "answer": "Yes"
        },
        {
            "passage": "I took my dog to the vet today and the doctor said she had a serious infection. He put her on antibiotics.",
            "question": "Did the vet prescribe antibiotics?",
            "answer": "Yes"
        },
        {
            "passage": "The new exhibit at the art museum is amazing. I would highly recommend it to anyone who likes modern art.",
            "question": "Would you recommend the museum exhibit?",
            "answer": "Yes"
        },
        {
            "passage": "The new technology has revolutionized the way businesses operate. It has made communication and collaboration easier than ever.",
            "question": "Has the new technology made communication and collaboration easier?",
            "answer": "Yes"
        },
        {
            "passage": "John studied for hours, but he still failed the math test. He was so disappointed.",
            "question": "Did John pass the math test?",
            "answer": "No"
        },
        {
            "passage": "The police officer was very kind and understanding when I got pulled over for speeding. He let me off with a warning.",
            "question": "Did the police officer give the driver a ticket?",
            "answer": "No"
        },
        {
            "passage": "My boss was so impressed with my presentation that he gave me a raise. I was so happy.",
            "question": "Did the boss give the employee a raise?",
            "answer": "Yes"
        },
        {
            "passage": "I decided to go for a run, even though it was raining. It was an invigorating experience.",
            "question": "Was it raining when the person went for a run?",
            "answer": "Yes"
        }
    ]).sample(frac=1).reset_index(drop=True),
    pd.DataFrame([
        {
            "passage": "Jack recommends his least favorite books of the year to his followers. The least favorite book this year was Harry Potter and the 7 Rings.",
            "question": "What book does Jack dislike?",
            "answer": "Jack does not like Harry Potter and the 7 Rings."
        },
        {
            "passage": "When Judy and Jack went to school, they got in trouble with their teacher for being late. I didn't think it was very fair.",
            "question": "Did she think it was fair?",
            "answer": "No, she didn't think it was very fair."
        },
        {
            "passage": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
            "question": "Is inflation good for society?",
            "answer": "Hmm. Do you think so?"
        },
        {
            "passage": "Put yourself out there. The more time you spend dating and socializing, the more likely you will find a boyfriend you like.",
            "question": "Does socializing help you find a boyfriend?",
            "answer": "Yes, it helps you find a boyfriend."
        },
        {
            "passage": "John was a great basketball player. He was always the one to make the winning shot.",
            "question": "What did John do well?",
            "answer": "John was a great basketball player and he was always the one to make the winning shot."
        },
        {
            "passage": "The government is trying to make it easier for people to get healthcare. They are increasing funding and expanding access.",
            "question": "What is the government doing to help with healthcare?",
            "answer": "The government is increasing funding and expanding access to make it easier for people to get healthcare."
        },
        {
            "passage": "George was always a great friend. He was always there when I needed someone to talk to.",
            "question": "What did George do for you?",
            "answer": "George was always a great friend and he was always there when I needed someone to talk to."
        },
        {
            "passage": "The new law states that all drivers must have car insurance. It is important for everyone to have the proper coverage.",
            "question": "What does the law say about car insurance?",
            "answer": "The law states that all drivers must have car insurance. It is important for everyone to have the proper coverage."
        },
        {
            "passage": "John was always so kind and generous. He was always willing to lend a helping hand.",
            "question": "What was John like?",
            "answer": "John was always so kind and generous. He was always willing to lend a helping hand."
        },
        {
            "passage": "The weather is expected to be sunny and hot for the next few days. Make sure to bring sunscreen if you plan on going out.",
            "question": "What type of weather is expected?",
            "answer": "The weather is expected to be sunny and hot for the next few days."
        },
        {
            "passage": "The new iPhone 11 Pro has a lot of great features. It has an improved camera and a faster processor.",
            "question": "What are some of the features of the iPhone 11 Pro?",
            "answer": "The iPhone 11 Pro has an improved camera and a faster processor."
        }
    ]).sample(frac=1).reset_index(drop=True),
    pd.DataFrame([
        {
            "passage": "Anna's mother always told her to be confident even if she feels nervous on the inside",
            "question": "Does Anna always feel nervous on the inside?",
            "answer": "Unknown"
        },
        {
            "passage": "Max and Jeff were extremely competitive at soccer, but Max was a lot better.",
            "question": "Was Jeff better than Max at soccer?",
            "answer": "No, Max was a lot better"
        },
        {
            "passage": "When Judy and Jack went to school, they got in trouble with their teacher for being late. I didn't think it was very fair.",
            "question": "Did she think it was fair?",
            "answer": "No, she didn't think it was very fair."
        },
        {
            "passage": "The FSP conference took place last week in Spain and representatives from 21 countries attended.",
            "question": "Did representatives from more than 20 countries attend FSP?",
            "answer": "Yes"
        },
        {
            "passage": "The students had to read the book before the end of the semester in order to get a good grade.",
            "question": "What did the students have to do to get a good grade?",
            "answer": "They had to read the book before the end of the semester."
        },
        {
            "passage": "John and his friends decided to take a road trip to a nearby beach. They were all very excited.",
            "question": "Were John and his friends excited about the road trip?",
            "answer": "Yes, they were all very excited."
        },
        {
            "passage": "Leslie was a very talented artist and she enjoyed creating artwork in her free time.",
            "question": "Did Leslie enjoy creating artwork?",
            "answer": "Yes, she enjoyed creating artwork in her free time."
        },
        {
            "passage": "The new bridge opened this week and the community was very excited about it.",
            "question": "Was the community excited about the new bridge?",
            "answer": "Yes, the community was very excited about it."
        },
        {
            "passage": "Brian's parents were very strict and never allowed him to stay out late.",
            "question": "Did Brian's parents allow him to stay out late?",
            "answer": "No, they never allowed him to stay out late."
        },
        {
            "passage": "The students had to take a test before the end of the semester in order to pass the class.",
            "question": "What did the students have to do to pass the class?",
            "answer": "They had to take a test before the end of the semester."
        }
    ]).sample(frac=1).reset_index(drop=True),
]

class CBDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, data_train, boost_id):
        return [
            questioner_examples[boost_id],
            openended_qa_examples[boost_id],
        ]

    def zero_few_baseline(
        self,
        test_data,
        few_shot_df,
        manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer,
        do_few_shot=True,
    ):
        expt_log = {}
        preds = []
        labels = []

        labels_names = set(test_data["targets_pretokenized"])
        labels_names = [l.lower().strip() for l in labels_names]

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if ind in expt_log:
                pred = entry["pred"]
                gold = entry["gold"]
            else:
                text = row["inputs_pretokenized"]
                gold = row["targets_pretokenized"]

                        

                text = row["inputs_pretokenized"]
                text = text.replace("True, False, or Neither?", "")
                text = text + ". True, False, or Neither?"
                gold = row["targets_pretokenized"]

                
                icl_str = ""
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        current_example = f"{s_row['inputs_pretokenized']} {s_row['targets_pretokenized']}\n\n\n"
                        buffer_token = 30
                        if len(tokenizer.encode(icl_str + current_example + text, truncation=False)) + buffer_token >= self.max_seq_len:
                            break                          
                        icl_str += current_example                     
                
                prompt = f"{icl_str}{{text:}}"
                pmp = prompt.format(text=text)
                if i == 0:
                    print(pmp)

                raw_answer = get_response(
                    pmp,
                    manifest_question,
                    overwrite=bool(overwrite_manifest_question),
                    max_toks=30,
                )
                answer = raw_answer.strip().lower()
                answer = answer.split("\n")
                answer = [a for a in answer if a]
                answer = [
                    a
                    for a in answer
                    if any(l.lower() in a.lower() for l in labels_names)
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
                is_maybe = "neither" in answer.split() or "unknown" in answer.split() or "maybe" in answer.split()
                pred = "Neither"
                if is_yes and (not is_maybe and not is_no):
                    pred = "True"
                if is_no and (not is_maybe and not is_yes):
                    pred = "False"
                if is_maybe and (not is_no and not is_yes):
                    pred = "Neither"
                entry = {
                    "ind": ind,
                    "example": text,
                    "base_prompt": pmp,
                    "raw_answer": raw_answer,
                    "pred": pred,
                    "gold": gold,
                }
                expt_log[ind] = entry

            preds.append(pred)
            labels.append(gold)

        report = classification_report(labels, preds, output_dict=True)
        return expt_log, report["accuracy"]

    def get_question(self, statement, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        quesiton_prompt = f"\n{prompt_suffix}\n\nStatement: {{statement:}}\nQuestion:"
        question_pmp = quesiton_prompt.format(statement=statement)
        answer = get_response(
            question_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        answer = answer.split("\n")[0]
        return answer, question_pmp

    def open_qa(self, question, passage, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        qa_prompt = f"\n{prompt_suffix}\n\nPassage: {{passage:}}\nQuestion: {question}\nAnswer:"
        qa_pmp = qa_prompt.format(passage=passage)
        answer = get_response(
            qa_pmp,
            manifest,
            overwrite=bool(overwrite_manifest),
            max_toks=50,
        )
        answer = answer.split("\n")[0]
        answer = (
            answer.replace("A: ", "")
            .replace("B: ", "")
            .replace("Answer: ", "")
            .replace(", ", " ")
        )
        return answer, qa_pmp

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer
    ):
    
        import random
        import numpy as np
        seed = 1000
        random.seed(seed)
        np.random.seed(seed)
        
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=-1)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train)
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            individual_accuracies.append(classification_report(labels, [p[i] for p in all_boost_preds], output_dict=True)["accuracy"])
        report = classification_report(labels, preds, output_dict=True)
        return expt_log, expt_log_train, report["accuracy"], individual_accuracies

    def _run_decomp_single_data(self, test_data, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=-1):
        expt_log = {}
        all_boost_preds = []
        labels = []

        # ################
        # import os
        # import json
        # model_name_question = os.environ['EXP_MODE_QUESTION']
        # # question_file = '/nvmedata/changranh/ama_question_synthetic_data/' + model_name_question + self.task_name + ".jsonl"
        # question_file = '/scratch/changranh/ama_question_synthetic_data/' + model_name_question + self.task_name + ".jsonl"        
        # ################  
        
        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            suffix = "True, False, or Neither?"
            input = row["inputs_pretokenized"]
            passage = input.split("\nQuestion: ")[0]
            statement = (
                input.split("\nQuestion: ")[-1]
                .replace(suffix, "")
                .replace('"', "")
                .strip()
            )

            if i == run_limit:
                break

            gold = row["targets_pretokenized"]
            gold = gold.lower()
            prompts_across_boost = []
            preds_across_boost = []
            for boost_examples in boost_dfs:
                all_prompts = []
                question, question_final_prompt = self.get_question(
                    statement, questioner, boost_examples[0], manifest_question, overwrite_manifest_question
                )
                open_answer, answer_final_prompt = self.open_qa(
                    question, passage, openended_qa, boost_examples[1], manifest_answer, overwrite_manifest_answer
                )
                all_prompts.append(question_final_prompt)
                all_prompts.append(answer_final_prompt)
                if i == 0:
                    print("\n".join(all_prompts))
                if "Yes" in open_answer.split():
                    answer = "True"
                elif "No" in open_answer.split():
                    answer = "False"
                else:
                    answer = "Neither"

                answer = answer.lower()
                
                is_yes = "yes" in answer.split() or "true" in answer.split()
                is_no = "no" in answer.split() or "false" in answer.split()
                is_maybe = "neither" in answer.split() or "maybe" in answer.split() or "unknown" in answer.split()
                pred = "neither"
                if is_yes and (not is_maybe and not is_no):
                    pred = "true"
                if is_no and (not is_maybe and not is_yes):
                    pred = "false"
                if is_maybe and (not is_no and not is_yes):
                    pred = "neither"
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)

            entry = {
                "ind": ind,
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "example": input,
                "gold": gold,
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(gold)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 3
    task_name = "super_glue_cb"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_cb_GPT_3_style/"
    decomp = CBDecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
