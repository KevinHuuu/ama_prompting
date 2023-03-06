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
# question_file = '/scratch/changranh/ama_question_synthetic_data/' + model_name_question + '_ANLIR3' + ".jsonl"        
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
            "statement": "The dog barked all night.",
            "question": "Did the dog bark all night?",
        },
        {
            "statement": "The car was broken.",
            "question": "Was the car broken?",
        },
        {
            "statement": "The house was painted yellow.",
            "question": "Was the house painted yellow?",
        },
        {
            "statement": "The teacher gave us a test.",
            "question": "Did the teacher give us a test?",
        },
        {
            "statement": "The students studied for the exam.",
            "question": "Did the students study for the exam?",
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
            "statement": "She has a beautiful smile.",
            "question": "Does she have a beautiful smile?",
        },
        {
            "statement": "The sky is blue.",
            "question": "Is the sky blue?",
        },
        {
            "statement": "The cat meowed loudly.",
            "question": "Did the cat meow loudly?",
        },
        {
            "statement": "The bus was late.",
            "question": "Was the bus late?",
        },
        {
            "statement": "The store closes at 8pm.",
            "question": "Does the store close at 8pm?"
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
            "statement": "The dog barked loudly at the mailman.",
            "question": "Did the dog bark at the mailman?",
        },
        {
            "statement": "I'm going to the store to buy some food.",
            "question": "Are you going to the store to buy some food?",
        },
        {
            "statement": "He was late to school because of the traffic.",
            "question": "Was he late to school because of the traffic?",
        },
        {
            "statement": "She loves to read books in her free time.",
            "question": "Does she love to read books in her free time?",
        },
        {
            "statement": "The mountain was too tall to climb.",
            "question": "Was the mountain too tall to climb?",
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
            "statement": "The cat loves to chase its tail.",
            "question": "Does the cat love to chase its tail?",
        },
        {
            "statement": "The store is closed on Sundays.",
            "question": "Is the store closed on Sundays?",
        },
        {
            "statement": "The moon is round.",
            "question": "Is the moon round?",
        },
        {
            "statement": "Eating healthy is good for you.",
            "question": "Is eating healthy good for you?",
        },
        {
            "statement": "He went to the store.",
            "question": "Did he go to the store?"
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
            "context": "The human brain is composed of about 100 billion neurons, and each neuron is connected to about 10,000 other neurons.",
            "question": "Based on the context, How many neurons are in the human brain? ",
            "answer": "100 billion neurons",
        },
        {
            "context": "The theory of evolution states that all living organisms are descended from a common ancestor.",
            "question": "Based on the context, Does the theory of evolution state that life had a single source of origin? ",
            "answer": "Yes, a common ancestor",
        },
        {
            "context": "The average temperature on Earth is 15 degrees Celsius.",
            "question": "Based on the context, What is the average temperature on Earth? ",
            "answer": "15 degrees Celsius",
        },
        {
            "context": "The ozone layer protects the Earth from the sun's ultraviolet radiation.",
            "question": "Based on the context, Does the ozone layer absorb ultraviolet radiation?  ",
            "answer": "Yes, it absorbs ultraviolet radiation",
        },
        {
            "context": "The Earth is estimated to be about 4.5 billion years old.",
            "question": "Based on the context, How old is the Earth? ",
            "answer": "4.5 billion years old",
        },
        {
            "context": "Gravity is the force that holds the planets in their orbits around the sun.",
            "question": "Based on the context, Does gravity affect the planets' orbits? ",
            "answer": "Yes, gravity affects the planets' orbits",
        },
        {
            "context": "The sun is composed primarily of hydrogen and helium.",
            "question": "Based on the context, What are the main components of the sun? ",
            "answer": "hydrogen and  helium",
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
            "context": "The Nile River is the longest river in the world, stretching 4,132 miles from its source in Burundi to its mouth in Egypt.",
            "question": "Based on the context, What is the source of the Nile River?",
            "answer": "Burundi"
        },
        {
            "context": "The term 'social stratification' refers to the way in which people are divided into different classes or hierarchical levels in society.",
            "question": "Based on the context, Does social stratification involve a division of labor?",
            "answer": "Yes"
        },
        {
            "context": "The theory of evolution states that all life is related and has descended from a common ancestor.",
            "question": "Based on the context, Is the theory of evolution supported by scientific evidence?",
            "answer": "Yes"
        },
        {
            "context": "The Milky Way is a barred spiral galaxy, about 100,000 light-years across, containing 200-400 billion stars.",
            "question": "Based on the context, What type of galaxy is the Milky Way?",
            "answer": "Barred spiral"
        },
        {
            "context": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides.",
            "question": "Based on the context, Is the hypotenuse always the longest side of the triangle?",
            "answer": "Yes"
        },
        {
            "context": "The theory of relativity explains the behavior and motion of objects in the universe and states that the speed of light is a constant.",
            "question": "Based on the context, Is the speed of light affected by gravity?",
            "answer": "Yes"
        },
        {
            "context": "The atom is the smallest unit of matter that still retains all the properties of an element.",
            "question": "Based on the context, Is a molecule smaller than an atom?",
            "answer": "Yes"
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
            "context": "Jackson went to the store to buy some cereal and a new video game.",
            "question": "Based on the context, What did Jackson go to the store to buy?",
            "answer": "Cereal and a new video game"
        },
        {
            "context": "The Smiths have been living in their house for 10 years.",
            "question": "Based on the context, How long have the Smiths been living in their house?",
            "answer": "10 years"
        },
        {
            "context": "The fireman was able to save the dog from the burning house.",
            "question": "Based on the context, What did the fireman save?",
            "answer": "The dog"
        },
        {
            "context": "The city council approved the new bridge project to span over the river.",
            "question": "Based on the context, What was the new project approved by the city council?",
            "answer": "A new bridge project"
        },
        {
            "context": "The mayor of the city held a press conference to announce the new policy.",
            "question": "Based on the context, What did the mayor announce at the press conference?",
            "answer": "A new policy"
        },
        {
            "context": "The students in Mrs. Smith's class are learning about the different types of plants and animals.",
            "question": "Based on the context, What are the students in Mrs. Smith's class learning about?",
            "answer": "Different types of plants and animals"
        },
        {
            "context": "The family went to the park for a picnic and to play some sports.",
            "question": "Based on the context, What did the family do at the park?",
            "answer": "Have a picnic and play sports"
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
            "context": "A study of the benefits of practicing mindfulness showed that participants felt more positive emotions, better relationships, and improved overall well-being.",
            "question": "Based on the context, Does mindfulness improve well-being?",
            "answer": "Yes"
        },
        {
            "context": "Developing countries are unable to fight poverty and maintain economic growth due to inadequate resources, infrastructure, and poverty-related factors.",
            "question": "Based on the context, Do inadequate resources and infrastructure hinder economic growth?",
            "answer": "Yes"
        },
        {
            "context": "The risk of cyberattacks is increasing, with a growing number of connected devices and an increase in the sophistication of attackers.",
            "question": "Based on the context, Is the risk of cyberattacks increasing?",
            "answer": "Yes"
        },
        {
            "context": "The sun is composed primarily of hydrogen and helium, which account for about 75% and 24% of its total mass respectively.",
            "question": "Based on the context, Does the sun consist mostly of hydrogen and helium?",
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
            "context": "The Big Bang Theory explains the origin of the universe and its expansion.",
            "question": "Based on the context, Does the Big Bang Theory explain the origin of the universe?",
            "answer": "yes"
        },
        {
            "context": "The British empire was the largest empire in history, spanning 5 continents and ruling over a quarter of the world's population.",
            "question": "Based on the context, Was the British empire the largest empire in history?",
            "answer": "yes"
        },
        {
            "context": "Albert Einstein developed the Theory of Relativity, which changed the way scientists think about the universe.",
            "question": "Based on the context, Did Albert Einstein develop the Theory of Relativity?",
            "answer": "yes"
        },
        {
            "context": "The Industrial Revolution saw a massive increase in technology, leading to the mechanization of factories and the growth of cities.",
            "question": "Based on the context, Was there an increase in technology during the Industrial Revolution?",
            "answer": "yes"
        },
        {
            "context": "The world's population has grown exponentially since the start of the 20th century, from 1.6 billion in 1900 to 7.6 billion in 2018.",
            "question": "Based on the context, Has the world's population grown since the start of the 20th century?",
            "answer": "yes"
        },
        {
            "context": "The Romans were known for their engineering prowess, creating aqueducts and roads that spanned the length and breadth of their empire.",
            "question": "Based on the context, Were the Romans known for their engineering prowess?",
            "answer": "yes"
        },
        {
            "context": "The internet has revolutionized communication, making it easier to stay in touch and access information.",
            "question": "Based on the context, Has the internet revolutionized communication?",
            "answer": "yes"
        }
    ]),
]

# ##############################################################################################################################
# # All prompts
# questioner_prompt = InputOutputPrompt(
#     input_formatter=lambda x: f"Statement: {x['statement']}",
#     output_formatter=lambda x: f"Question: {x['question']}",
#     required_keys=["question", "statement"],
#     input_output_sep="\n",
#     example_sep="\n\n",
#     instruction="Rewrite the statement as a yes/no question.\n\n"
# )
# questioner_prompt_examples = [
#     pd.DataFrame([
#         {
#             "statement": "most of the light comes from the sun",
#             "question": "Does most of the light come from the sun?"
#         },
#         {
#             "statement": "the test was not hard",
#             "question": "Was the test not hard?",
#         },
#         {
#             "statement": "it is a good idea to buy your parents gifts",
#             "question": "Is it a good idea to buy your parents gifts?",
#         },
#         {
#             "statement": "the balloon popped",
#             "question": "Did the balloon pop?",
#         },
#         {
#             "statement": "The father and son went camping to California.",
#             "question": "Did the father and son go camping?",
#         },
#     ]),
#     pd.DataFrame([
#         {
#             "statement": "most of the light comes from the sun",
#             "question": "Does most of the light come from the sun?"
#         },
#         {
#             "statement": "the test was not",
#             "question": "Was the test not hard?",
#         },
#         {
#             "statement": "it is a good idea to buy your parents gifts",
#             "question": "Is it a good idea to buy your parents gifts?",
#         },
#         {
#             "statement": "the balloon popped",
#             "question": "Did the balloon pop?",
#         },
#         {
#             "statement": "The father and son went camping to California.",
#             "question": "Did the father and son go camping?",
#         },
#     ]),
#     pd.DataFrame([
#         {
#             "statement": "most of the light comes from the sun",
#             "question": "Does most of the light come from the sun?"
#         },
#         {
#             "statement": "the test was not hard",
#             "question": "Was the test not hard?",
#         },
#         {
#             "statement": "it is a good idea to buy your parents gifts",
#             "question": "Is it a good idea to buy your parents gifts?",
#         },
#         {
#             "statement": "the balloon popped",
#             "question": "Did the balloon pop?",
#         },
#         {
#             "statement": "The father and son went camping to California.",
#             "question": "Did the father and son go camping?",
#         },
#     ]),
#     pd.DataFrame([
#         {
#             "statement": "most of the light comes from the sun",
#             "question": "Does most of the light come from the sun?"
#         },
#         {
#             "statement": "the test was not hard",
#             "question": "Was the test not hard?",
#         },
#         {
#             "statement": "it is a good idea to buy your parents gifts",
#             "question": "Is it a good idea to buy your parents gifts?",
#         },
#         {
#             "statement": "the balloon popped",
#             "question": "Did the balloon pop?",
#         },
#         {
#             "statement": "The father and son went camping to California.",
#             "question": "Did the father and son go camping?",
#         },
#     ]),
#     pd.DataFrame([
#         {
#             "statement": "most of the light comes from the sun",
#             "question": "Does most of the light come from the sun?"
#         },
#         {
#             "statement": "the test was not hard",
#             "question": "Was the test not hard?",
#         },
#         {
#             "statement": "it is a good idea to buy your parents gifts",
#             "question": "Is it a good idea to buy your parents gifts?",
#         },
#         {
#             "statement": "the balloon popped",
#             "question": "Did the balloon pop?",
#         },
#         {
#             "statement": "The father and son went camping to California.",
#             "question": "Did the father and son go camping?",
#         },
#     ]),
# ]

# extraction_qa = InputOutputPrompt(
#     input_formatter=lambda x: f"Context: {x['context']}\nQuestion: {x['question']}",
#     output_formatter=lambda x: f"Answer: {x['answer']}",
#     required_keys=["context", "question", "answer"],
#     input_output_sep="\n",
#     example_sep="\n\n",
#     instruction="Answer the question. If there is no evidence in the context, return \"Unknown\".\n\n"
# )
# extraction_qa_examples = [
#     pd.DataFrame([
#         {
#             "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
#             "question": "Based on the context, Did the plague affect people in Europe?",
#             "answer": "yes, people in Italy, Europe",
#         },
#         {
#             "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
#             "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
#             "answer": "unknown",
#         },
#         {
#             "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
#             "question": "Based on the context, Is anti-matter made of electrons? ",
#             "answer": "Unknown",
#         },
#     ]),
#     pd.DataFrame([
#         {
#             "context": "According to Biraben, the plague was present somewhere in Italy only between 1346 and 1671, and not after that.",
#             "question": "Based on the context, Was the plague present in Italy during the 2000s?",
#             "answer": "No, it was present between 1346 and 1671"
#         },
#         {
#             "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
#             "question": "Based on the context, Is anti-matter made of electrons? ",
#             "answer": "Unknown"
#         },
#         {
#             "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
#             "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
#             "answer": "unknown"
#         }
#     ]),
#     pd.DataFrame([
#         {
#             "context": "Jenna's 10th birthday was yesterday evening and at least 10 of her friends attended the party.",
#             "question": "Based on the context, Did 10 friends attend Jenna's party?",
#             "answer": "Unknown"
#         },
#         {
#             "context": "The bullies attacked John when he was walking through the elementary school parking lot and then got sent to the teacher's office.",
#             "question": "Based on the context, Did the bullies attack John in the teacher's office?",
#             "answer": "No, parking lot"
#         },
#         {
#             "context": "WISS discovered a new monkey disease occurring in a remote tribe in the Amazon rainforrest.",
#             "question": "Based on the context, Did WISS discover a new monkey species?",
#             "answer": "No, a new monkey disease"
#         }
#     ]),
#     pd.DataFrame([
#         {
#             "context": "When Judy and Jack went to school, they got in trouble with their teacher for being late. I didn't think it was very fair.",
#             "question": "Based on the context, Did she think it was fair?",
#             "answer": "No"
#         },
#         {
#             "context": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
#             "question": "Based on the context, Is inflation good for society?",
#             "answer": "Unknown"
#         },
#         {
#             "context": "Put yourself out there. The more time you spend dating and socializing, the more likely you will find a boyfriend you like.",
#             "question": "Based on the context, Does socializing help you find a boyfriend?",
#             "answer": "Yes"
#         },
#         {
#             "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
#             "question": "Based on the context, Did the plague affect people in Europe?",
#             "answer": "yes, people in Italy, Europe",
#         },
#         {
#             "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
#             "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
#             "answer": "unknown",
#         },
#         {
#             "context": "The term \"matter\" is used throughout physics in a bewildering variety of contexts: for example, one refers to \"condensed matter physics\", \"elementary matter\", \"partonic\" matter, \"dark\" matter, \"anti\"-matter, \"strange\" matter, and \"nuclear\" matter.",
#             "question": "Based on the context, Is anti-matter made of electrons? ",
#             "answer": "Unknown",
#         },
#     ]),
#     pd.DataFrame([
#          {
#             "context": "According to Biraben, the plague was present somewhere in Italy and affected 1,200 people.",
#             "question": "Based on the context, Did the plague affect over 1,000 people?",
#             "answer": "yes, 1,200 people",
#         },
#         {
#             "context": "If inflation is occurring, leading to higher prices for basic necessities such as gas by 2 dollars. Do you think that inflation is good for society?",
#             "question": "Based on the context, Is inflation good for society?",
#             "answer": "Unknown"
#         },
#         {
#             "context": "Policies aiming at controlling unemployment and in particular at reducing its inequality-associated effects support economic growth.",
#             "question": "Based on the context, Is confidence a factor in increasing self-esteem?",
#             "answer": "unknown"
#         }
#     ]),
# ]

class ANLIDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_few_shot_examples(self, train_data, k_shot):
        """Get few shot examples"""
        labels = [' False', ' True', ' Neither']
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
        labels = [' True', ' False', ' Neither']
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
                text = row["inputs_pretokenized"]
                text = text.replace("True, False, or Neither?", "").strip().strip("\n")
                text = text + " True, False, or Neither? "
                gold = row["targets_pretokenized"]
                
                icl_str = ""                
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        current_example = f"{s_row['inputs_pretokenized']}{s_row['targets_pretokenized']}\n\n"
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
        else:
            answer = ''
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
    task_name = "anli_r3"
    args.num_boost = 5
    data_dir = f"{DATA_DIR}/P3/data_feather/anli_GPT_3_style_r3"
    decomp = ANLIDecomp(task_name, data_dir, val_split="test")
    decomp.run(args)


if __name__ == "__main__":
    main()
