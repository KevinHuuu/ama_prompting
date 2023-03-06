#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import sys
import json
import string
import datetime



################# Copy paste from /nvmedata/changranh/fm_in_context_eval_data_for_fewshot_data_collection/realtimeqa_public/scripts/utils/tools.py
import json, jsonlines, datetime, string
from collections import Counter

from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def read_jsonl(in_file):
    questions = []
    with open(in_file) as fin:
        for line in fin:
            question = json.loads(line)
            questions.append(question)
    return questions

def fall_back(data_1, data_2, top_k=5):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]
        datum_1["search_result"] = [article for article in datum_1["search_result"] if "text" in article]
        nb_retrieved = len(datum_1["search_result"])
        if nb_retrieved < top_k:
            datum_1["search_result"].extend(datum_2["search_result"][:top_k-nb_retrieved])
    return data_1

def answer2jsonl(answers, questions, out_file, scores=None):
    # Confirm we have answers for all questions
    assert len(answers) == len(questions)
    if scores is not None:
        assert len(answers) == len(scores)
    outputs = []
    for q_idx in range(len(answers)):
        if scores is None:
            output = {"question_id": questions[q_idx]["question_id"], "prediction": answers[q_idx]}
        else:
            output = {"question_id": questions[q_idx]["question_id"], "prediction": answers[q_idx], "score": scores[q_idx]}
        outputs.append(output)
    with jsonlines.open(out_file, mode='w') as fout:
        fout.write_all(outputs)

def wiki2jsonl(questions, retrieved_docs, out_file):
    outputs = []
    assert len(questions) == len(retrieved_docs)
    for question, retrieved in zip(questions, retrieved_docs):
        search_result = []
        for doc_id, doc, doc_scores, publish_date in zip(retrieved["doc_ids"], retrieved["docs"], retrieved["doc_scores"], retrieved["publish_dates"]):
            search_result.append({"doc_id": doc_id, "text": doc, "doc_scores": doc_scores, "publish_date": publish_date})
        output = {"question_id": question["question_id"], "search_result": search_result}
        outputs.append(output)
    with jsonlines.open(out_file, mode='w') as fout:
        fout.write_all(outputs)

def check_jsonls(data_1, data_2):
    assert len(data_1) == len(data_2)
    for datum_1, datum_2 in zip(data_1, data_2):
        assert datum_1["question_id"] == datum_2["question_id"]

def add_today(sentence, date):
    date = datetime.datetime.strptime(date, '%Y/%m/%d')
    date = date.strftime("%B %d, %Y")
    sentence = "Today is {}. ".format(date) + sentence
    return sentence

def normalize_answer(s):
    # TODO: should we keep those counter removal? 
    def remove_counter(text):
        return text.replace("年", "").replace("歳", "").replace("人", "").replace("년", "")

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_counter(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

#########################
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

realtime_qa_path = Path(f"{DATA_DIR}/realtimeqa_public/scripts/utils")
print("realtime_qa_path:", realtime_qa_path)
sys.path.insert(0,str(realtime_qa_path))

try:
    from tools import f1_score, metric_max_over_ground_truths, fall_back, read_jsonl, check_jsonls
except ModuleNotFoundError:
    print(f"realtimeQA tools not found. Please download from realtimeQA repo to {realtime_qa_path}.")


art1_answer = InputOutputPrompt(
    input_formatter=lambda x: f"{x['passages']}Question: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["passages", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer the question given the article. Answer \"I don't know\" if you don't know.",
)

all_answer = InputOutputPrompt(
    input_formatter=lambda x: f"{x['passages']}Question: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["passages", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Answer the question given the articles."
)

art1_answer_examples = [
    pd.DataFrame([
        {
            "passages": "Article 1: Walmart is slashing prices on clothing and other products - CNN New York(CNN Business) Many shoppers have pulled back on buying clothing and other discretionary items as the highest inflation in four decades pinches their pocketbooks.",
            "question": "Which major US retailer announced this week it is slashing prices on clothing and other products?",
            "answer": "\"Walmart\""
        },
        {
            "passages": "Article 1: Oak Fire: California's fast-moving wildfire burns 14,000 acres and ... (CNN) A wildfire raging for a third day Sunday in central California's Mariposa County outside Yosemite National Park has burned more than 14, 000 acres and forced thousands to evacuate from rural communities.",
            "question": "A raging wildfire this week forced thousands of people to evacuate communities near which national park?",
            "answer": "\"Yosemite National Park\""
        },
        {
            "passages": "Article 1: Frontier Airlines, Spirit Airlines announce budget airline merger Budget airlines Frontier Airlines and Spirit Airlines.",
            "question": "Which airline announced a deal this week to buy Spirit Airlines?",
            "answer": "\"I don't know\""
        },
        {
            "passages": "Article 1: Apple launches new line of Macbooks Apple Inc. launched its latest line of Macbooks on Thursday, featuring faster processors and improved graphics.",
            "question": "Which technology giant announced their latest line of Macbooks this week?",
            "answer": "\"Apple\""
        },
        {
            "passages": "Article 1: US reports record number of new coronavirus cases in single day - NBC News NBC News) The US reported a record number of new coronavirus cases on Thursday, with over 75,000 Americans testing positive for the virus in a single day, according to Johns Hopkins University.",
            "question": "How many coronavirus cases were reported in the US in a single day this week?",
            "answer": "\"75,000\""
        },
        {
            "passages": "Article 1: Amazon to double its workforce in 2020 Amazon said on Thursday that it plans to double its workforce in 2020, adding an additional 500,000 new jobs.",
            "question": "How many new jobs is Amazon planning to create this year?",
            "answer": "\"500,000\""
        },
        {
            "passages": "Article 1: Netflix announces new streaming service Netflix announced on Tuesday that it will launch a new streaming service focused on original content.",
            "question": "Which streaming giant announced a new service this week?",
            "answer": "\"Netflix\""
        },
        {
            "passages": "Article 1: Disney announces plans to reopen theme parks Disney said on Tuesday that it is planning to reopen its theme parks in Florida and California in July with enhanced safety protocols.",
            "question": "Which company announced plans this week to reopen its theme parks?",
            "answer": "\"Disney\""
        },
        {
            "passages": "Article 1: White House unveils new economic stimulus package The White House unveiled its new economic stimulus package on Wednesday, with $1 trillion in spending to help businesses and individuals affected by the coronavirus pandemic.",
            "question": "Which government body announced an economic stimulus package this week?",
            "answer": "\"White House\""
        },
        {
            "passages": "Article 1: Facebook announces new shopping platform Facebook announced on Thursday the launch of a new shopping platform, allowing users to browse and buy products from their favorite brands and stores directly on the social media platform.",
            "question": "Which social media giant announced a new shopping platform this week?",
            "answer": "\"Facebook\""
        }
    ]).sample(frac=1).reset_index(drop=True),
    pd.DataFrame([
        {
            "passages": "Article 1: During the initial outbreak in Wuhan, the virus and disease were commonly referred to as \"coronavirus\", \"Wuhan coronavirus\", \"the coronavirus outbreak\" and the \"Wuhan coronavirus outbreak\", with the disease sometimes called \"Wuhan pneumonia\".",
            "question": "From which country did COVID originate?",
            "answer": "\"Wuhan, China\""
        },
        {
            "passages": "Article 1: Philippines earthquake: 7.0-magnitude quake in Abra kills five.",
            "question": "Which country was shaken by a 7.0-magnitude earthquake this week?",
            "answer": "\"Philippines\""
        },
        {
            "passages": "Article 1: Ten Things You Need to Know Today: 10 August 2022 | The Week UK Domino’s fails in land of the pizza Domino’s Pizza has been forced to pull out of the home of the pizza.",
            "question": "What percentage of babies in the UK were born out of wedlock last year?",
            "answer": "\"I don't know\""
        },
        {
            "passages": "Article 1: The European Union and the United Kingdom have agreed to a Brexit trade deal.",
            "question": "What is the name of the Brexit trade deal?",
            "answer": "\"The EU-UK Trade and Cooperation Agreement\""
        },
        {
            "passages": "Article 1: The Coronavirus pandemic has caused an unprecedented global health crisis.",
            "question": "What is the cause of the global health crisis?",
            "answer": "\"The Coronavirus pandemic\""
        },
        {
            "passages": "Article 1: France is still in the process of vaccinating its population against the coronavirus.",
            "question": "What is the name of the virus France is vaccinating against?",
            "answer": "\"Coronavirus\""
        },
        {
            "passages": "Article 1: India has recently launched a mission to send its first manned spacecraft to the moon.",
            "question": "What is the name of India's mission to send a manned spacecraft to the moon?",
            "answer": "\"Gaganyaan\""
        },
        {
            "passages": "Article 1: California has recently implemented a new law banning the sale of flavored e-cigarettes.",
            "question": "Which US state has recently implemented a ban on flavored e-cigarettes?",
            "answer": "\"California\""
        },
        {
            "passages": "Article 1: The United Nations has recently warned about the risk of food insecurity in the wake of the coronavirus pandemic.",
            "question": "What institution has recently warned about the risk of food insecurity due to the coronavirus pandemic?",
            "answer": "\"The United Nations\""
        },
        {
            "passages": "Article 1: Pope Francis has recently visited Iraq in the first papal visit in over 20 years.",
            "question": "Who recently visited Iraq on the first papal visit in over 20 years?",
            "answer": "\"Pope Francis\""
        }
    ]).sample(frac=1).reset_index(drop=True),
    pd.DataFrame([
        {
            "passages": "Article 1: Japan\'s Sakurajima volcano erupts, prompting evacuation alerts.",
            "question": "A volcano eruption in which country recently prompted a Level 5 alert – the highest level – calling for people to evacuate.?",
            "answer": "\"Japan\""
        },
        {
            "passages": "Article 1: \'Weekends With Adele\': Caesars Palace Las Vegas to host residency Adele has her fans exclaiming \"Oh My God\" with her latest announcement.",
            "question": "Which popular singer will launch a Las Vegas residency later this year?",
            "answer": "\"Adele\""
        },
        {
            "passages": "Article 1: As of 2020, the largest selling record label in the U.S. is Sony Music Entertainment.",
            "question": "Which artist is best seller 2020?",
            "answer": "\"I don't know\""
        },
        {
            "passages": "Article 1: The Nobel Prize in Physics 2020 was awarded to Roger Penrose, Reinhard Genzel and Andrea Ghez.",
            "question": "Who was awarded the Nobel Prize in Physics in 2020?",
            "answer": "\"Roger Penrose, Reinhard Genzel and Andrea Ghez\""
        },
        {
            "passages": "Article 1: The US presidential election is scheduled for November 3, 2020.",
            "question": "When is the US presidential election?",
            "answer": "\"November 3, 2020\""
        },
        {
            "passages": "Article 1: The 2020 Summer Olympics in Tokyo have been postponed to 2021 due to the Covid-19 pandemic.",
            "question": "When have the 2020 Summer Olympics been rescheduled to?",
            "answer": "\"2021\""
        },
        {
            "passages": "Article 1: The 2020 World Series was won by the Los Angeles Dodgers.",
            "question": "Which team won the 2020 World Series?",
            "answer": "\"Los Angeles Dodgers\""
        },
        {
            "passages": "Article 1: The 2020 Golden Globe Awards were presented on Sunday, February 28.",
            "question": "When were the 2020 Golden Globe Awards presented?",
            "answer": "\"February 28\""
        },
        {
            "passages": "Article 1: The 2020 Academy Awards were held on Sunday, February 9.",
            "question": "When were the 2020 Academy Awards held?",
            "answer": "\"February 9\""
        },
        {
            "passages": "Article 1: The 2020 US presidential election was won by Joe Biden.",
            "question": "Who won the 2020 US presidential election?",
            "answer": "\"Joe Biden\""
        },
        {
            "passages": "Article 1: Pope Francis is the current Bishop of Rome and head of the Catholic Church.",
            "question": "Who is the current head of the Catholic Church?",
            "answer": "\"Pope Francis\""
        }
    ]).sample(frac=1).reset_index(drop=True)
]
all_answer_examples = [
    pd.DataFrame([
        {
            "passages": "Article 1: Walmart is slashing prices on clothing and other products - CNN New York(CNN Business) Many shoppers have pulled back on buying clothing and other discretionary items as the highest inflation in four decades pinches their pocketbooks.\nArticle 2: Retail slowdown: Target cuts vendor orders, slashes prices as it ... Associated Press NEW YORK.\nArticle 3: Stores have too much stuff. That means discounts are coming | CNN ... New York(CNN Business).\nArticle 4: GM reports strong sales but says it's prepared for possible recession ... New York (CNN Business).\nArticle 5: Target is ramping up discounts. Here's why - CNN New York(CNN Business).\n",
            "question": "Which major US retailer announced this week it is slashing prices on clothing and other products?",
            "answer": "\"Walmart\""
        },
        {
            "passages": "Article 1: Article 1: JetBlue announces a deal to buy Spirit Airlines. Fares could surge.\nArticle 2: JetBlue-Spirit merger: Airlines have complaints over flights and fees Christopher Elliott Special to USA TODAY.\nArticle 3: JetBlue announces a deal to buy Spirit Airlines | CNN Business The announcement comes a day after Spirit pulled the plug on a deal to merge with Frontier.\nArticle 4: Spirit and Frontier pull plug on deal, setting stage for JetBlue to buy ... New York (CNN Buiness).\nArticle 5: Frontier Airlines, Spirit Airlines announce budget airline merger Budget airlines Frontier Airlines and Spirit Airlines.\n",
            "question": "Which airline announced a deal this week to buy Spirit Airlines?",
            "answer": "\"JetBlue\""
        },
        {
            "passages": "Article 1: Oak Fire: California's fast-moving wildfire burns 14,000 acres and ... (CNN) A wildfire raging for a third day Sunday in central California's Mariposa County outside Yosemite National Park has burned more than 14, 000 acres and forced thousands to evacuate from rural communities.\nArticle 2: California Oak Fire: Rapidly-growing fire engulfs homes near ... For more on the fires, \" United Shades of America with W. Kamau Bell \" heads to California to discover how communities are learning to coexist with the frequent destruction.\nArticle 3: 5 things to know for July 25: Wildfires, Ukraine, Monkeypox, Volcano ... If your day doesn't start until you're up to speed on the latest headlines, then let us introduce you to your new favorite morning fix.\nArticle 4: Wildfires in US: 2 firefighting helicopter pilots die in Idaho ... Multiple wildfires raged across the U.S. Saturday, causing deaths, destruction and thousands of forced evacuations.\nArticle 5: Boulder wildfires: Hundreds of homes burn evacuations ordered BOULDER, Colo. — A ferocious wind-driven wildfire on Thursday destroyed hundreds of homes and businesses near Denver, forcing tens of thousands to flee and blanketing the area in smoke.\n",
            "question": "A raging wildfire this week forced thousands of people to evacuate communities near which national park?",
            "answer": "\"Yosemite National Park\""
        },
        {
            "passages": "Article 1: Facebook and Instagram to ban ads promoting anti-vaccine misinformation ... New York (CNN Business) Facebook and Instagram announced Thursday they will begin removing ads promoting anti-vaccine misinformation.\nArticle 2: Facebook says it will stop running anti-vaccine ads Reuters Facebook Inc said on Thursday it would stop running ads containing anti-vaccine misinformation on its platforms.\nArticle 3: Facebook bans anti-vaccine misinformation from its platforms | CNN Business Facebook and Instagram said Thursday they will begin removing ads promoting anti-vaccine misinformation.\nArticle 4: Facebook and Instagram to block anti-vaccine ads | Fox Business Facebook and Instagram are cracking down on ads that promote anti-vaccine misinformation.\nArticle 5: Facebook and Instagram to ban ads promoting anti-vaccine misinformation ... New York (CNN Business) Facebook and Instagram announced Thursday they will begin removing ads promoting anti-vaccine misinformation.\n",
            "question": "Which social media platforms have announced they will begin removing ads promoting anti-vaccine misinformation?",
            "answer": "\"Facebook and Instagram\""
        },
        {
            "passages": "Article 1: Starbucks is planning to open 3,000 new stores in China by 2022 ... CNN New York(CNN Business) Starbucks is planning to open more than 3,000 new stores in mainland China by 2022, the company announced Thursday.\nArticle 2: Starbucks plans to open 3,000 stores in China over the next three years ... CNBC Starbucks is planning to open 3,000 new stores in mainland China over the next three years.\nArticle 3: Starbucks to open 3,000 more stores in China by 2022 | CNN Business Starbucks is planning to open more than 3,000 new stores in mainland China by 2022.\nArticle 4: Starbucks Announces Plans to Open 3,000 New Stores in China by 2022 ... The Wall Street Journal Starbucks plans to open more than 3,000 new stores in mainland China by 2022.\nArticle 5: Starbucks plans to open 3,000 stores in China over the next three years | CNBC Starbucks is planning to open 3,000 new stores in mainland China over the next three years.\n",
            "question": "How many new stores is Starbucks planning to open in mainland China by 2022?",
            "answer": "\"More than 3,000\""
        }
    ]).sample(frac=1).reset_index(drop=True),
    pd.DataFrame([
        {
            "passages": "Article 1: During the initial outbreak in Wuhan, the virus and disease were commonly referred to as \"coronavirus\", \"Wuhan coronavirus\", \"the coronavirus outbreak\" and the \"Wuhan coronavirus outbreak\", with the disease sometimes called \"Wuhan pneumonia\".\nArticle 2: The first known outbreak started in Wuhan, Hubei, China, in November 2019.\nArticle 3: A cluster of patients in China\’s Hubei Province, in the city of Wuhan, begin to experience the symptoms of an atypical pneumonia-like illness that does not respond well to standard treatments.\nArticle 4: The World Health Organization(WHO) has released its plan to investigate the origins of the COVID pandemic. The search will start in Wuhan.\nArticle 5: The World Health Organization(WHO) Country Office in China is informed of several cases of a pneumonia of unknown etiology(cause) with symptoms including shortness of breath and fever occurring in Wuhan, China.\n",
            "question": "From which country did COVID originate?",
            "answer": "\"Wuhan, China\""
        },
        {
            "passages": "Article 1: Philippines earthquake: 7.0-magnitude quake in Abra kills five.\nArticle 2: Haiti earthquakes: Comparing recent quake to deadly 2010 tragedy A decade ago, an earthquake struck just outside Haiti's capital of Port-au-Prince.\nArticle 3: Indonesia earthquake: Death toll rises as Lombok, Bali shaken The death toll rose to 98 after a magnitude 7.0 earthquake rocked the Indonesian island of Lombok and nearby Bali.\nArticle 4: Alaska earthquake: Aftershocks continue to shake Last Frontier Two days after a magnitude 7.0 earthquake struck near Anchorage, Alaska is still shaking.\n",
            "question": "Which country was shaken by a 7.0-magnitude earthquake this week?",
            "answer": "\"Philippines\""
        },
        {
            "passages": "Article 1: According to latest Office for National Statistics (ONS) data, of 624,828 live births registered, 320,713 were to women who were not married or in a civil partnership at the time – 51.3% of the total.\nArticle 2: Ten Things You Need to Know Today: 10 August 2022 | The Week UK Domino’s fails in land of the pizza Domino’s Pizza has been forced to pull out of the home of the pizza.\nArticle 3: Gay couple sues State Department for denying daughter's citizenship Couple's daughter was born to a surrogate in Britain.\nArticle 4: Ex-Jehovah's Witnesses say church's shunning caused too many.\nArticle 5: Kids before marriage is becoming the norm (and that\'s not good) What\’s wrong with America? Everybody has an answer these days.\n",
            "question": "What percentage of babies in the UK were born out of wedlock last year?",
            "answer": "\"51.3%\""
        },
        {
            "passages": "Article 1: The United Nations Educational, Scientific and Cultural Organization (UNESCO) has designated 851 World Heritage Sites around the world.\nArticle 2: The Taj Mahal, a white marble mausoleum located in Agra, India, is a UNESCO World Heritage Site.\nArticle 3: The Great Wall of China, an ancient Chinese fortification built to protect the country from invaders, is a UNESCO World Heritage Site.\nArticle 4: The Grand Canyon, a vast canyon located in northern Arizona in the United States, is a UNESCO World Heritage Site.\nArticle 5: Machu Picchu, an ancient Incan fortress located in the Andes Mountains of Peru, is a UNESCO World Heritage Site.\n",
            "question": "How many UNESCO World Heritage Sites are there in the world?",
            "answer": "\"851\""
        }
    ]).sample(frac=1).reset_index(drop=True),
    pd.DataFrame([
        {
            "passages": "Article 1: Japan\'s Sakurajima volcano erupts, prompting evacuation alerts.\nArticle 2: (CNN) Here\'s a tip if you are among the millions of people quitting your job: Don\'t cash out your 401(k)! As tempting as it may seem, there are other options that will give you better returns in the long run.\nArticle 3: (CNN) Did you take your vitamins this morning? Daily vitamin D and fish oil supplements may help prevent some adults from developing autoimmune disorders such as arthritis and psoriasis.\nArticle 4: (CNN) The federal tax filing season is underway. And even though the IRS still hasn\'t processed millions of returns from last year due to Covid-19 and a lack of funding, there are still ways to help ensure your tax filing experience is hassle-free.\nArticle 5: (CNN) Happy Valentine\'s Day, and happy Conveniently Call In Sick to Work Day for all the Los Angeles Rams fans out there.\n",
            "question": "A volcano eruption in which country recently prompted a Level 5 alert – the highest level – calling for people to evacuate.?",
            "answer": "\"Japan\""
        },
        {
            "passages": "Article 1: \'Weekends With Adele\': Caesars Palace Las Vegas to host residency Adele has her fans exclaiming \"Oh My God\" with her latest announcement.\nArticle 2: Usher is \'ready to drop\' upcoming album \'Confessions 2\' this year Associated Press LOS ANGELES — Usher has a confession — he’s almost ready to release the sequel to his groundbreaking, epic 2004 album “Confessions.”\nArticle 3: When could Britney Spears start her Vegas residency shows in 2022?\nArticle 4: Backstreet Boys return to Las Vegas for holiday residency Backstreet will be back in Las Vegas.\nArticle 5: Miami Vice postpones Las Vegas residency due to rising COVID risks.\n",
            "question": "Which popular singer will launch a Las Vegas residency later this year?",
            "answer": "\"Adele\""
        },
        {
            "passages": "Article 1: Best selling artists worldwide and Eminem makes into the Top 10. You can check the lit below: 1 BTS 2 Taylor Swift 3 Drake 4 The Weeknd 5 Billie Eilish.\nArticle 2: Pop group BTS have been named the number one artists of 2020 by the IFPI.\nArticle 3: The market reported total revenues of $21.6 billion (roughly €18.2bn), marking its sixth consecutive year of growth and the highest figure since 2002.\nArticle 4: BTS were the world\'s biggest act overall, marking the first time a South Korean band has topped the global chart.\nArticle 5: Music is an art form, and cultural activity, whose medium is sound. Well, music has no language and is soothing and stress relieving, no matter in whichever language it is.\n",
            "question": "Which artist is best seller 2020?",
            "answer": "\"pop group BTS\""
        },
        {
            "passages": "Article 1: U.S. President Joe Biden has proposed a plan to send four relief checks of $1,400 to Americans in 2021.\nArticle 2: Biden\'s plan, which comes as part of a $1.9 trillion stimulus package, includes an additional $1,400 in direct payments for individuals who make less than $75,000 annually.\nArticle 3: The payments, which would come after the $600 payments sent out in December, will be sent to individuals making less than $75,000 and couples making less than $150,000.\nArticle 4: The plan also includes an extension of unemployment benefits and other aid, such as an increase in the child tax credit.\nArticle 5: \"The proposal we\'re announcing today is about giving the backbone of this nation – the essential workers, the working people, the people who built the country – a fighting chance,\" Biden said in a speech on Thursday.\n",
            "question": "How much will individuals making less than $75,000 annually receive in the proposed relief checks?",
            "answer": "\"$1,400\""
        }
    ]).sample(frac=1).reset_index(drop=True)
]
# Taken from realtimeQA github repo
# https://github.com/realtimeqa/realtimeqa_public
def get_retrieved_text(retrieved_datum, top_k=5, rm_date_r=False):
    search_result = retrieved_datum["search_result"]
    retrieved_text = ""
    for art_i, article in enumerate(search_result[:top_k]):
        if "publish_date" not in article:
            continue
        date = article["publish_date"]
        content = article["text"]
        if content == '':
            continue
        date = datetime.datetime.strptime(date, '%Y/%m/%d')
        date = date.strftime("%B %d, %Y")
        #first_paraph = content.split("\n\n")[0]
        first_paraph = " ".join(content.split("\n\n")[:2])
        if "title" in article.keys():
            first_paraph = article["title"] + " " + first_paraph
        if not rm_date_r:
            retrieved_text += "Article on {}: {}\n".format(date, first_paraph)
        else:
            retrieved_text += "Article: {}\n".format(first_paraph)
    return retrieved_text

def read_dates(data_dir, dates):
    all_dfs = []
    for date in dates:
        passages = []
        in_file = str(Path(data_dir) / f"{date}_qa.jsonl")
        gold_df = pd.DataFrame([json.loads(line) for line in open(in_file)])
        gold_df["gold_answers"] = gold_df.apply(lambda x: [x["choices"][int(idx)] for idx in x["answer"]], axis=1)

        gcs_file = in_file.replace("_qa.jsonl", "_gcs.jsonl")
        dpr_file = in_file.replace("_qa.jsonl", "_dpr.jsonl")
        gcs = read_jsonl(gcs_file)
        dpr = read_jsonl(dpr_file)
        check_jsonls(gcs, dpr)
        retrieved_data = fall_back(gcs, dpr)
        for q_idx in range(len(gold_df)):
            retrieved_text = get_retrieved_text(retrieved_data[q_idx], top_k=5, rm_date_r=True)
            passages.append(retrieved_text)
        gold_df["passages"] = passages
        all_dfs.append(gold_df)
    return pd.concat(all_dfs)

class RealtimeQADecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, train_data, boost_id):
        # We boost by changing passage order
        return [
            art1_answer_examples[boost_id],
            all_answer_examples[boost_id]
        ]

    def read_data(self, save_dir, overwrite_data):
        val_dates_for_benchmark = ["20220617", "20220624", "20220701", "20220708", "20220715", "20220722"]
        train_dates_for_benchmark = ["20220729", "20220805", "20220812"]
        save_data = Path(f"{save_dir}/{self.task_name}/data.jsonl")
        if not save_data.exists() or overwrite_data:
            test_data = read_dates(self.data_dir, val_dates_for_benchmark)
            test_data = test_data.reset_index(drop=True)
            test_data.to_feather(f"{save_data}")
        else:
            test_data = pd.read_feather(f"{save_data}")

        save_data = Path(f"{save_dir}/{self.task_name}/train_data.feather")
        if not save_data.exists() or overwrite_data:
            train_data = read_dates(self.data_dir, train_dates_for_benchmark)
            train_data = train_data.reset_index(drop=True)
            train_data.to_feather(f"{save_data}")
        else:
            train_data = pd.read_feather(f"{save_data}")

        print(f"Test Data Size: {len(test_data)}")
        print(f"Train Data Size: {len(train_data)}")
        return test_data, train_data

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

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):  
            try:
                row["answer"] = row["answer"].tolist()
                row["choices"] = row["choices"].tolist()
                row["gold_answers"] = row["gold_answers"].tolist()
            except:
                pass
            question = row["question_sentence"]
            passages = row["passages"]
            golds = row["gold_answers"]
            assert len(golds) == 1
            icl_str = ""
            if do_few_shot:                                   
                # Taken from realtime_qa github repo
                current_example = f"Question: What is the capital city of Japan?\nAnswer: Tokyo\n\n"
                buffer_token = 30
                if len(tokenizer.encode(icl_str + current_example + passages + question, truncation=False)) + buffer_token >= self.max_seq_len:
                    pass
                else:
                    icl_str += current_example                                              

            pmp = f"{icl_str}{passages}Question: {question}\nAnswer:"
            
            if i <= 3:
                print("########icl_str")
                print(icl_str)
                print('########pmp')
                print(pmp)
                print("########end")   
                
                
            if i == 0:
                print(pmp)
            try:
                raw_answer = get_response(
                    pmp,
                    manifest_answer,
                    overwrite=bool(overwrite_manifest_answer),
                    max_toks=30,
                )
            except:
                passage_list = [p for p in passages.split("Article:") if p.strip()]
                passage_list = [" ".join(p.split(" ")[:100]) for p in passage_list]
                passages = "Article:" + "Article:".join(passage_list)
                pmp = f"{passages}Question: {question}\nAnswer:"
                # raw_answer = get_response(
                #     pmp,
                #     manifest_answer,
                #     overwrite=bool(overwrite_manifest_answer),
                #     max_toks=30,
                # )
                
                ##################################
                raw_answer = 'placeholder yes no yes no'
                ##################################                 
                
                
                
            pred = raw_answer.split("\n")[0].strip()

            entry = {
                "ind": ind,
                "prompt": pmp.strip(),
                "completion": " " + golds[0].strip().capitalize(),
            }
            expt_log[ind] = entry

            preds.append(pred)
            labels.append(golds[0])
        metric = np.mean([metric_max_over_ground_truths(f1_score, pred, [gold]) for pred, gold in zip(preds, labels)])
        # Compute accuracy
        # metric = np.mean([pred == gold for pred, gold in zip(preds, labels)])
        return expt_log, metric


    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=1000)
        # Do WS
        preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train)
        # Get accuracies across all boost sets
        individual_accuracies = []
        for i in range(len(all_boost_preds[0])):
            individual_accuracies.append(
                np.mean([metric_max_over_ground_truths(f1_score, pred, [gold]) for pred, gold in zip([p[i] for p in all_boost_preds], labels)])
            )
        metric = np.mean([metric_max_over_ground_truths(f1_score, pred, [gold]) for pred, gold in zip(preds, labels)])
        return expt_log, expt_log_train, metric, individual_accuracies

    def _run_decomp_single_data(self, test_data, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=-1):
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
            if i == run_limit:
                break
            row["answer"] = [d for d in row["answer"]]
            row["choices"] = [d for d in row["choices"]]
            row["gold_answers"] = [d for d in row["gold_answers"]]

            question = row["question_sentence"]
            passages = row["passages"]
            golds = row["gold_answers"]
            assert len(golds) == 1

            prompts_across_boost = []
            preds_across_boost = []
            for boost_idx, boost_examples in enumerate(boost_dfs):
                all_prompts = []
                
                passage_list = [p for p in passages.split("Article:") if p.strip()]
                passage_list = [" ".join(p.split(" ")[:100]) for p in passage_list]
                assert len(passage_list) > 1
                passages_for_prompt = "" .join([f"Article {i+1}:{a}" for i, a in enumerate(passage_list[:1])])
                # Art1 answer
                icl_str = art1_answer(boost_examples[0])
                pmp = f"{icl_str}\n\n{passages_for_prompt}Question: {question}\nAnswer:"
                all_prompts.append(pmp)
                raw_answer_art1 = get_response(
                    pmp,
                    manifest_answer,
                    overwrite=bool(overwrite_manifest_answer),
                    max_toks=30,
                )
                pred_art1 = raw_answer_art1.split("\n")[0].strip("\"").strip()

                icl_str = all_answer(boost_examples[1])
                passages_for_prompt = "" .join([f"Article {i+1}:{a}" for i, a in enumerate(passage_list)])
                all_pmp = f"{icl_str}\n\n{passages_for_prompt}Question: {question}\nAnswer:"
                all_prompts.append(all_pmp)

                if i == 0:
                    print(pmp)
                    print(all_pmp)

                raw_answer_all = get_response(
                    all_pmp,
                    manifest_answer,
                    overwrite=bool(overwrite_manifest_answer),
                    max_toks=30,
                )
                pred_all = raw_answer_all.split("\n")[0].strip("\"").strip()
                if pred_art1 == "I don't know":
                    pred = pred_all
                else:
                    pred = pred_art1

                pred = pred.translate(str.maketrans('', '', string.punctuation))
                pred = pred.lower()
                # if pred != golds[0].lower() and golds[0].lower() in passages_for_prompt.lower():
                #     print("PASSAGES", passages_for_prompt)
                #     print("QUESTION", question)
                #     print("ANSWER", golds[0])
                #     print("PRED", pred)
                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)
            
            entry = {
                "ind": ind,
                "example": row.to_dict(),
                "prompts": prompts_across_boost,
                "preds_boost": preds_across_boost,
                "gold": golds[0].lower(),
            }
            expt_log[ind] = entry
            all_boost_preds.append(preds_across_boost)
            labels.append(golds[0])
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    if not Path(realtime_qa_path).exists():
        raise ValueError(f"Path {realtime_qa_path} does not exist. Download from realtimeQA repo to this path.")
    task_name = "realtime_qa"
    data_dir = f"{DATA_DIR}/realtimeqa_public/past/2022"
    if not Path(data_dir).exists():
        raise ValueError(f"Data dir {data_dir} does not exist. Download from realtimeQA repo.")
    decomp = RealtimeQADecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()


# #!/usr/bin/env python
# # coding: utf-8
# from pathlib import Path
# from tqdm.auto import tqdm
# import pandas as pd
# import numpy as np
# import sys
# import json
# import string
# import datetime

# from decomposition import Decomposition, get_args, DATA_DIR
# from utils import get_response, InputOutputPrompt

# realtime_qa_path = Path(f"{DATA_DIR}/realtimeqa_public/scripts/utils")
# sys.path.insert(0,str(realtime_qa_path))
# try:
#     from tools import f1_score, metric_max_over_ground_truths, fall_back, read_jsonl, check_jsonls
# except ModuleNotFoundError:
#     print(f"realtimeQA tools not found. Please download from realtimeQA repo to {realtime_qa_path}.")


# art1_answer = InputOutputPrompt(
#     input_formatter=lambda x: f"{x['passages']}Question: {x['question']}",
#     output_formatter=lambda x: f"Answer: {x['answer']}",
#     required_keys=["passages", "question", "answer"],
#     input_output_sep="\n",
#     example_sep="\n\n",
#     instruction="Answer the question given the article. Answer \"I don't know\" if you don't know.",
# )

# all_answer = InputOutputPrompt(
#     input_formatter=lambda x: f"{x['passages']}Question: {x['question']}",
#     output_formatter=lambda x: f"Answer: {x['answer']}",
#     required_keys=["passages", "question", "answer"],
#     input_output_sep="\n",
#     example_sep="\n\n",
#     instruction="Answer the question given the articles."
# )

# art1_answer_examples = [
#     pd.DataFrame([
#         {
#             "passages": "Article 1: Walmart is slashing prices on clothing and other products - CNN New York(CNN Business) Many shoppers have pulled back on buying clothing and other discretionary items as the highest inflation in four decades pinches their pocketbooks.",
#             "question": "Which major US retailer announced this week it is slashing prices on clothing and other products?",
#             "answer": "\"Walmart\""
#         },
#         {
#             "passages": "Article 1: Oak Fire: California's fast-moving wildfire burns 14,000 acres and ... (CNN) A wildfire raging for a third day Sunday in central California's Mariposa County outside Yosemite National Park has burned more than 14, 000 acres and forced thousands to evacuate from rural communities.",
#             "question": "A raging wildfire this week forced thousands of people to evacuate communities near which national park?",
#             "answer": "\"Yosemite National Park\""
#         },
#         {
#             "passages": "Article 1: Frontier Airlines, Spirit Airlines announce budget airline merger Budget airlines Frontier Airlines and Spirit Airlines.",
#             "question": "Which airline announced a deal this week to buy Spirit Airlines?",
#             "answer": "\"I don't know\""
#         }
#     ]),
#     pd.DataFrame([
#         {
#             "passages": "Article 1: During the initial outbreak in Wuhan, the virus and disease were commonly referred to as \"coronavirus\", \"Wuhan coronavirus\", \"the coronavirus outbreak\" and the \"Wuhan coronavirus outbreak\", with the disease sometimes called \"Wuhan pneumonia\".",
#             "question": "From which country did COVID originate?",
#             "answer": "\"Wuhan, China\""
#         },
#         {
#             "passages": "Article 1: Philippines earthquake: 7.0-magnitude quake in Abra kills five.",
#             "question": "Which country was shaken by a 7.0-magnitude earthquake this week?",
#             "answer": "\"Philippines\""
#         },
#         {
#             "passages": "Article 1: Ten Things You Need to Know Today: 10 August 2022 | The Week UK Domino’s fails in land of the pizza Domino’s Pizza has been forced to pull out of the home of the pizza.",
#             "question": "What percentage of babies in the UK were born out of wedlock last year?",
#             "answer": "\"I don't know\""
#         }
#     ]),
#     pd.DataFrame([
#         {
#             "passages": "Article 1: Japan\'s Sakurajima volcano erupts, prompting evacuation alerts.",
#             "question": "A volcano eruption in which country recently prompted a Level 5 alert – the highest level – calling for people to evacuate.?",
#             "answer": "\"Japan\""
#         },
#         {
#             "passages": "Article 1: \'Weekends With Adele\': Caesars Palace Las Vegas to host residency Adele has her fans exclaiming \"Oh My God\" with her latest announcement.",
#             "question": "Which popular singer will launch a Las Vegas residency later this year?",
#             "answer": "\"Adele\""
#         },
#         {
#             "passages": "Article 1: As of 2020, the largest selling record label in the U.S. is Sony Music Entertainment.",
#             "question": "Which artist is best seller 2020?",
#             "answer": "\"I don't know\""
#         }
#     ])
# ]
# all_answer_examples = [
#     pd.DataFrame([
#         {
#             "passages": "Article 1: Walmart is slashing prices on clothing and other products - CNN New York(CNN Business) Many shoppers have pulled back on buying clothing and other discretionary items as the highest inflation in four decades pinches their pocketbooks.\nArticle 2: Retail slowdown: Target cuts vendor orders, slashes prices as it ... Associated Press NEW YORK.\nArticle 3: Stores have too much stuff. That means discounts are coming | CNN ... New York(CNN Business).\nArticle 4: GM reports strong sales but says it's prepared for possible recession ... New York (CNN Business).\nArticle 5: Target is ramping up discounts. Here's why - CNN New York(CNN Business).\n",
#             "question": "Which major US retailer announced this week it is slashing prices on clothing and other products?",
#             "answer": "\"Walmart\""
#         },
#         {
#             "passages": "Article 1: Article 1: JetBlue announces a deal to buy Spirit Airlines. Fares could surge.\nArticle 2: JetBlue-Spirit merger: Airlines have complaints over flights and fees Christopher Elliott Special to USA TODAY.\nArticle 3: JetBlue announces a deal to buy Spirit Airlines | CNN Business The announcement comes a day after Spirit pulled the plug on a deal to merge with Frontier.\nArticle 4: Spirit and Frontier pull plug on deal, setting stage for JetBlue to buy ... New York (CNN Buiness).\nArticle 5: Frontier Airlines, Spirit Airlines announce budget airline merger Budget airlines Frontier Airlines and Spirit Airlines.\n",
#             "question": "Which airline announced a deal this week to buy Spirit Airlines?",
#             "answer": "\"JetBlue\""
#         },
#         {
#             "passages": "Article 1: Oak Fire: California's fast-moving wildfire burns 14,000 acres and ... (CNN) A wildfire raging for a third day Sunday in central California's Mariposa County outside Yosemite National Park has burned more than 14, 000 acres and forced thousands to evacuate from rural communities.\nArticle 2: California Oak Fire: Rapidly-growing fire engulfs homes near ... For more on the fires, \" United Shades of America with W. Kamau Bell \" heads to California to discover how communities are learning to coexist with the frequent destruction.\nArticle 3: 5 things to know for July 25: Wildfires, Ukraine, Monkeypox, Volcano ... If your day doesn't start until you're up to speed on the latest headlines, then let us introduce you to your new favorite morning fix.\nArticle 4: Wildfires in US: 2 firefighting helicopter pilots die in Idaho ... Multiple wildfires raged across the U.S. Saturday, causing deaths, destruction and thousands of forced evacuations.\nArticle 5: Boulder wildfires: Hundreds of homes burn evacuations ordered BOULDER, Colo. — A ferocious wind-driven wildfire on Thursday destroyed hundreds of homes and businesses near Denver, forcing tens of thousands to flee and blanketing the area in smoke.\n",
#             "question": "A raging wildfire this week forced thousands of people to evacuate communities near which national park?",
#             "answer": "\"Yosemite National Park\""
#         }
#     ]),
#     pd.DataFrame([
#         {
#             "passages": "Article 1: During the initial outbreak in Wuhan, the virus and disease were commonly referred to as \"coronavirus\", \"Wuhan coronavirus\", \"the coronavirus outbreak\" and the \"Wuhan coronavirus outbreak\", with the disease sometimes called \"Wuhan pneumonia\".\nArticle 2: The first known outbreak started in Wuhan, Hubei, China, in November 2019.\nArticle 3: A cluster of patients in China\’s Hubei Province, in the city of Wuhan, begin to experience the symptoms of an atypical pneumonia-like illness that does not respond well to standard treatments.\nArticle 4: The World Health Organization(WHO) has released its plan to investigate the origins of the COVID pandemic. The search will start in Wuhan.\nArticle 5: The World Health Organization(WHO) Country Office in China is informed of several cases of a pneumonia of unknown etiology(cause) with symptoms including shortness of breath and fever occurring in Wuhan, China.\n",
#             "question": "From which country did COVID originate?",
#             "answer": "\"Wuhan, China\""
#         },
#         {
#             "passages": "Article 1: Philippines earthquake: 7.0-magnitude quake in Abra kills five.\nArticle 2: Haiti earthquakes: Comparing recent quake to deadly 2010 tragedy A decade ago, an earthquake struck just outside Haiti's capital of Port-au-Prince.\nArticle 3: Indonesia earthquake: Death toll rises as Lombok, Bali shaken The death toll rose to 98 after a magnitude 7.0 earthquake rocked the Indonesian island of Lombok and nearby Bali.\nArticle 4: Alaska earthquake: Aftershocks continue to shake Last Frontier Two days after a magnitude 7.0 earthquake struck near Anchorage, Alaska is still shaking.\n",
#             "question": "Which country was shaken by a 7.0-magnitude earthquake this week?",
#             "answer": "\"Philippines\""
#         },
#         {
#             "passages": "Article 1: According to latest Office for National Statistics (ONS) data, of 624,828 live births registered, 320,713 were to women who were not married or in a civil partnership at the time – 51.3% of the total.\nArticle 2: Ten Things You Need to Know Today: 10 August 2022 | The Week UK Domino’s fails in land of the pizza Domino’s Pizza has been forced to pull out of the home of the pizza.\nArticle 3: Gay couple sues State Department for denying daughter's citizenship Couple's daughter was born to a surrogate in Britain.\nArticle 4: Ex-Jehovah's Witnesses say church's shunning caused too many.\nArticle 5: Kids before marriage is becoming the norm (and that\'s not good) What\’s wrong with America? Everybody has an answer these days.\n",
#             "question": "What percentage of babies in the UK were born out of wedlock last year?",
#             "answer": "\"51.3%\""
#         }
#     ]),
#     pd.DataFrame([
#         {
#             "passages": "Article 1: Japan\'s Sakurajima volcano erupts, prompting evacuation alerts.\nArticle 2: (CNN) Here\'s a tip if you are among the millions of people quitting your job: Don\'t cash out your 401(k)! As tempting as it may seem, there are other options that will give you better returns in the long run.\nArticle 3: (CNN) Did you take your vitamins this morning? Daily vitamin D and fish oil supplements may help prevent some adults from developing autoimmune disorders such as arthritis and psoriasis.\nArticle 4: (CNN) The federal tax filing season is underway. And even though the IRS still hasn\'t processed millions of returns from last year due to Covid-19 and a lack of funding, there are still ways to help ensure your tax filing experience is hassle-free.\nArticle 5: (CNN) Happy Valentine\'s Day, and happy Conveniently Call In Sick to Work Day for all the Los Angeles Rams fans out there.\n",
#             "question": "A volcano eruption in which country recently prompted a Level 5 alert – the highest level – calling for people to evacuate.?",
#             "answer": "\"Japan\""
#         },
#         {
#             "passages": "Article 1: \'Weekends With Adele\': Caesars Palace Las Vegas to host residency Adele has her fans exclaiming \"Oh My God\" with her latest announcement.\nArticle 2: Usher is \'ready to drop\' upcoming album \'Confessions 2\' this year Associated Press LOS ANGELES — Usher has a confession — he’s almost ready to release the sequel to his groundbreaking, epic 2004 album “Confessions.”\nArticle 3: When could Britney Spears start her Vegas residency shows in 2022?\nArticle 4: Backstreet Boys return to Las Vegas for holiday residency Backstreet will be back in Las Vegas.\nArticle 5: Miami Vice postpones Las Vegas residency due to rising COVID risks.\n",
#             "question": "Which popular singer will launch a Las Vegas residency later this year?",
#             "answer": "\"Adele\""
#         },
#         {
#             "passages": "Article 1: Best selling artists worldwide and Eminem makes into the Top 10. You can check the lit below: 1 BTS 2 Taylor Swift 3 Drake 4 The Weeknd 5 Billie Eilish.\nArticle 2: Pop group BTS have been named the number one artists of 2020 by the IFPI.\nArticle 3: The market reported total revenues of $21.6 billion (roughly €18.2bn), marking its sixth consecutive year of growth and the highest figure since 2002.\nArticle 4: BTS were the world\'s biggest act overall, marking the first time a South Korean band has topped the global chart.\nArticle 5: Music is an art form, and cultural activity, whose medium is sound. Well, music has no language and is soothing and stress relieving, no matter in whichever language it is.\n",
#             "question": "Which artist is best seller 2020?",
#             "answer": "\"pop group BTS\""
#         }
#     ])
# ]
# # Taken from realtimeQA github repo
# # https://github.com/realtimeqa/realtimeqa_public
# def get_retrieved_text(retrieved_datum, top_k=5, rm_date_r=False):
#     search_result = retrieved_datum["search_result"]
#     retrieved_text = ""
#     for art_i, article in enumerate(search_result[:top_k]):
#         if "publish_date" not in article:
#             continue
#         date = article["publish_date"]
#         content = article["text"]
#         if content == '':
#             continue
#         date = datetime.datetime.strptime(date, '%Y/%m/%d')
#         date = date.strftime("%B %d, %Y")
#         #first_paraph = content.split("\n\n")[0]
#         first_paraph = " ".join(content.split("\n\n")[:2])
#         if "title" in article.keys():
#             first_paraph = article["title"] + " " + first_paraph
#         if not rm_date_r:
#             retrieved_text += "Article on {}: {}\n".format(date, first_paraph)
#         else:
#             retrieved_text += "Article: {}\n".format(first_paraph)
#     return retrieved_text

# def read_dates(data_dir, dates):
#     all_dfs = []
#     for date in dates:
#         passages = []
#         in_file = str(Path(data_dir) / f"{date}_qa.jsonl")
#         gold_df = pd.DataFrame([json.loads(line) for line in open(in_file)])
#         gold_df["gold_answers"] = gold_df.apply(lambda x: [x["choices"][int(idx)] for idx in x["answer"]], axis=1)

#         gcs_file = in_file.replace("_qa.jsonl", "_gcs.jsonl")
#         dpr_file = in_file.replace("_qa.jsonl", "_dpr.jsonl")
#         gcs = read_jsonl(gcs_file)
#         dpr = read_jsonl(dpr_file)
#         check_jsonls(gcs, dpr)
#         retrieved_data = fall_back(gcs, dpr)
#         for q_idx in range(len(gold_df)):
#             retrieved_text = get_retrieved_text(retrieved_data[q_idx], top_k=5, rm_date_r=True)
#             passages.append(retrieved_text)
#         gold_df["passages"] = passages
#         all_dfs.append(gold_df)
#     return pd.concat(all_dfs)

# class RealtimeQADecomp(Decomposition):
#     def __init__(self, task_name, data_dir, val_split="validation"):
#         super().__init__(task_name, data_dir, val_split)

#     def get_boost_decomp_examples(self, train_data, boost_id):
#         # We boost by changing passage order
#         return [
#             art1_answer_examples[boost_id],
#             all_answer_examples[boost_id]
#         ]

#     def read_data(self, save_dir, overwrite_data):
#         val_dates_for_benchmark = ["20220617", "20220624", "20220701", "20220708", "20220715", "20220722"]
#         train_dates_for_benchmark = ["20220729", "20220805", "20220812"]
#         save_data = Path(f"{save_dir}/{self.task_name}/data.jsonl")
#         if not save_data.exists() or overwrite_data:
#             test_data = read_dates(self.data_dir, val_dates_for_benchmark)
#             test_data = test_data.reset_index(drop=True)
#             test_data.to_feather(f"{save_data}")
#         else:
#             test_data = pd.read_feather(f"{save_data}")

#         save_data = Path(f"{save_dir}/{self.task_name}/train_data.feather")
#         if not save_data.exists() or overwrite_data:
#             train_data = read_dates(self.data_dir, train_dates_for_benchmark)
#             train_data = train_data.reset_index(drop=True)
#             train_data.to_feather(f"{save_data}")
#         else:
#             train_data = pd.read_feather(f"{save_data}")

#         print(f"Test Data Size: {len(test_data)}")
#         print(f"Train Data Size: {len(train_data)}")
#         return test_data, train_data

#     def zero_few_baseline(
#         self,
#         test_data,
#         few_shot_df,
#         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer,
#         do_few_shot=True,
#     ):
#         expt_log = {}
#         preds = []
#         labels = []

#         for i, (ind, row) in tqdm(
#             enumerate(test_data.iterrows()), total=len(test_data)
#         ):  
#             try:
#                 row["answer"] = row["answer"].tolist()
#                 row["choices"] = row["choices"].tolist()
#                 row["gold_answers"] = row["gold_answers"].tolist()
#             except:
#                 pass
#             question = row["question_sentence"]
#             passages = row["passages"]
#             golds = row["gold_answers"]
#             assert len(golds) == 1
#             icl_str = ""
#             if do_few_shot:
#                 # Taken from realtime_qa github repo
#                 icl_str += f"Question: What is the capital city of Japan?\nAnswer: Tokyo\n\n"

#             pmp = f"{icl_str}{passages}Question: {question}\nAnswer:"
#             if i == 0:
#                 print(pmp)
#             try:
#                 raw_answer = get_response(
#                     pmp,
#                     manifest_answer,
#                     overwrite=bool(overwrite_manifest_answer),
#                     max_toks=30,
#                 )
#             except:
#                 passage_list = [p for p in passages.split("Article:") if p.strip()]
#                 passage_list = [" ".join(p.split(" ")[:100]) for p in passage_list]
#                 passages = "Article:" + "Article:".join(passage_list)
#                 pmp = f"{passages}Question: {question}\nAnswer:"
#                 raw_answer = get_response(
#                     pmp,
#                     manifest_answer,
#                     overwrite=bool(overwrite_manifest_answer),
#                     max_toks=30,
#                 )
#             pred = raw_answer.split("\n")[0].strip()

#             entry = {
#                 "ind": ind,
#                 "example": row.to_dict(),
#                 "base_prompt": pmp,
#                 "raw_answer": raw_answer,
#                 "pred": pred,
#                 "gold": golds[0],
#             }
#             expt_log[ind] = entry

#             preds.append(pred)
#             labels.append(golds[0])
#         metric = np.mean([metric_max_over_ground_truths(f1_score, pred, [gold]) for pred, gold in zip(preds, labels)])
#         # Compute accuracy
#         # metric = np.mean([pred == gold for pred, gold in zip(preds, labels)])
#         return expt_log, metric


#     def run_decomposed_prompt(
#         self, test_data, boost_data_train, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer
#     ):
#         expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer)
#         expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=1000)
#         # Do WS
#         preds = self.merge_boosted_preds(all_boost_preds, all_boost_train_preds, train_labels, expt_log, expt_log_train)
#         # Get accuracies across all boost sets
#         individual_accuracies = []
#         for i in range(len(all_boost_preds[0])):
#             individual_accuracies.append(
#                 np.mean([metric_max_over_ground_truths(f1_score, pred, [gold]) for pred, gold in zip([p[i] for p in all_boost_preds], labels)])
#             )
#         metric = np.mean([metric_max_over_ground_truths(f1_score, pred, [gold]) for pred, gold in zip(preds, labels)])
#         return expt_log, expt_log_train, metric, individual_accuracies

#     def _run_decomp_single_data(self, test_data, boost_dfs,         manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=-1):
#         expt_log = {}
#         all_boost_preds = []
#         labels = []


#         ################
#         import os
#         import json
#         model_name_question = os.environ['EXP_MODE_QUESTION']
#         # question_file = '/nvmedata/changranh/ama_question_synthetic_data/' + model_name_question + self.task_name + ".jsonl"
#         question_file = '/scratch/changranh/ama_question_synthetic_data/' + model_name_question + self.task_name + ".jsonl"        
#         ################  
        
#         for i, (ind, row) in tqdm(
#             enumerate(test_data.iterrows()), total=len(test_data)
#         ):
#             if i == run_limit:
#                 break
#             row["answer"] = [d for d in row["answer"]]
#             row["choices"] = [d for d in row["choices"]]
#             row["gold_answers"] = [d for d in row["gold_answers"]]

#             question = row["question_sentence"]
#             passages = row["passages"]
#             golds = row["gold_answers"]
#             assert len(golds) == 1

#             prompts_across_boost = []
#             preds_across_boost = []
#             for boost_idx, boost_examples in enumerate(boost_dfs):
#                 all_prompts = []
                
#                 passage_list = [p for p in passages.split("Article:") if p.strip()]
#                 passage_list = [" ".join(p.split(" ")[:100]) for p in passage_list]
#                 assert len(passage_list) > 1
#                 passages_for_prompt = "" .join([f"Article {i+1}:{a}" for i, a in enumerate(passage_list[:1])])
#                 # Art1 answer
#                 icl_str = art1_answer(boost_examples[0])
#                 pmp = f"{icl_str}\n\n{passages_for_prompt}Question: {question}\nAnswer:"
#                 all_prompts.append(pmp)
#                 raw_answer_art1 = get_response(
#                     pmp,
#                     manifest_answer,
#                     overwrite=bool(overwrite_manifest_answer),
#                     max_toks=30,
#                 )
#                 pred_art1 = raw_answer_art1.split("\n")[0].strip("\"").strip()

#                 icl_str = all_answer(boost_examples[1])
#                 passages_for_prompt = "" .join([f"Article {i+1}:{a}" for i, a in enumerate(passage_list)])
#                 all_pmp = f"{icl_str}\n\n{passages_for_prompt}Question: {question}\nAnswer:"
#                 all_prompts.append(all_pmp)

#                 if i == 0:
#                     print(pmp)
#                     print(all_pmp)

#                 raw_answer_all = get_response(
#                     all_pmp,
#                     manifest_answer,
#                     overwrite=bool(overwrite_manifest_answer),
#                     max_toks=30,
#                 )
#                 pred_all = raw_answer_all.split("\n")[0].strip("\"").strip()
#                 if pred_art1 == "I don't know":
#                     pred = pred_all
#                 else:
#                     pred = pred_art1

#                 pred = pred.translate(str.maketrans('', '', string.punctuation))
#                 pred = pred.lower()
#                 # if pred != golds[0].lower() and golds[0].lower() in passages_for_prompt.lower():
#                 #     print("PASSAGES", passages_for_prompt)
#                 #     print("QUESTION", question)
#                 #     print("ANSWER", golds[0])
#                 #     print("PRED", pred)
#                 prompts_across_boost.append(all_prompts)
#                 preds_across_boost.append(pred)
            
#             entry = {
#                 "ind": ind,
#                 "example": row.to_dict(),
#                 "prompts": prompts_across_boost,
#                 "preds_boost": preds_across_boost,
#                 "gold": golds[0].lower(),
#             }
#             expt_log[ind] = entry
#             all_boost_preds.append(preds_across_boost)
#             labels.append(golds[0])
#         return expt_log, all_boost_preds, labels


# def main():
#     args = get_args()
#     if not Path(realtime_qa_path).exists():
#         raise ValueError(f"Path {realtime_qa_path} does not exist. Download from realtimeQA repo to this path.")
#     task_name = "realtime_qa"
#     data_dir = f"{DATA_DIR}/realtimeqa_public/past/2022"
#     if not Path(data_dir).exists():
#         raise ValueError(f"Data dir {data_dir} does not exist. Download from realtimeQA repo.")
#     decomp = RealtimeQADecomp(task_name, data_dir)
#     decomp.run(args)


# if __name__ == "__main__":
#     main()
