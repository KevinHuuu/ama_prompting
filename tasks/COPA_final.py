#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
from collections import Counter
import pandas as pd
import random
import numpy as np

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

what_next = InputOutputPrompt(
    input_formatter=lambda x: f"Question: {x['example']}",
    output_formatter=lambda x: f"{x['continue']}",
    required_keys=["example", "continue"],
    input_output_sep=" ",
    example_sep="\n\n",
    instruction="Pick the correct ending for the example.\n\n"
)

what_next_examples = [
    pd.DataFrame([
        {
            "example": "(because 'she took medicine', because 'she got expelled') My roommate was feeling better because?",
            "continue": "'she took medicine'",
        },
        {
            "example": "(because 'he does not practice', because 'he is fast') Matt is not good at soccer because?", 
            "continue": "'he does not practice'",
        },
        {
            "example": "(because 'she was smart', because 'she never did her homework') The girl went to college and graduated with honors because?", 
            "continue": "'she was smart'",
        },
        {
            "example": "(because 'he did not listen', because 'he was arrogant') The student failed the class because?", 
            "continue": "'he did not listen'",
        },
        {
            "example": "(because 'he was lazy', because 'he was poor') The boy could not get a job because?", 
            "continue": "'he was lazy'",
        },
        {
            "example": "(because 'he was overweight', because 'he was healthy') The man was not accepted into the military because?", 
            "continue": "'he was overweight'",
        },
        {
            "example": "(because 'she was late', because 'she was early') The woman was not allowed to board the plane because?", 
            "continue": "'she was late'",
        },
        {
            "example": "(because 'he was not qualified', because 'he was not skilled') The man did not get the job because?", 
            "continue": "'he was not qualified'",
        },
        {
            "example": "(because 'he was not prepared', because 'he was not confident') The student failed the exam because?", 
            "continue": "'he was not prepared'",
        },
        {
            "example": "(because 'he was not honest', because 'he was not trustworthy') The man was not given the position because?", 
            "continue": "'he was not honest'",
        }
    ]),
    pd.DataFrame([
        {
            "example": "(so 'he is always tired', so 'he is always sleeping') My dad works very hard so",
            "continue": "'he is always tired'",
        },
        {
            "example": "(so 'she threw a party', so 'she took medicine') My roommate was sick so", 
            "continue": "'she took medicine'",
        },
        {
            "example": "(so 'he played', so 'he cried') Andy's parents got him a new toy so", 
            "continue": "'he played'",
        },
        {
            "example": "(so 'she shouted', so 'she smiled') Jenny saw her best friend so", 
            "continue": "'she smiled'",
        },
        {
            "example": "(so 'he won', so 'he celebrated') Joe entered a contest so", 
            "continue": "'he won'",
        },
        {
            "example": "(so 'he ate', so 'he cried') The baby was hungry so", 
            "continue": "'he ate'",
        },
        {
            "example": "(so 'it rained', so 'the flowers died') The garden didn't get enough sunlight so",
            "continue": "'the flowers died'",
        },
        {
            "example": "(so 'she sang', so 'she danced') The music was so loud so", 
            "continue": "'she sang'",
        },
        {
            "example": "(so 'he laughed', so 'he smiled') The joke was funny so", 
            "continue": "'he laughed'",
        },
        {
            "example": "(so 'he studied', so 'he passed') John wanted to get an A so", 
            "continue": "'he studied'",
        }
    ]),
]

question = InputOutputPrompt(
    input_formatter=lambda x: f"Question: {x['example']}",
    output_formatter=lambda x: f"Answer: {x['continue']}",
    required_keys=["example", "continue"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Pick the correct ending for the example.\n\n"
)

question_examples = [
    pd.DataFrame([
        {
            "example": "What best continues the sentence \"My dad often talks about long hours at work because\"?",
            "continue": "\"work is hard\"",
        },
        {
            "example": "What best continues the sentence \"My roommate was sick and took medicine and so\"?", 
            "continue": "\"she felt better\"",
        },
        {
            "example": "What best continues the sentence \"Andy's parents got him a new toy and so\"?",
            "continue": "\"he played\"",
        },
        {
            "example": "What best continues the sentence \"My roommate was feeling better because\"?",
            "continue": "\"she took medicine\"",
        },
        {
            "example": "What best continues the sentence \"My cat loves to play with string and so\"?",
            "continue": "\"she chases it around the house\"",
        },
        {
            "example": "What best continues the sentence \"My friend had a great day at school because\"?",
            "continue": "\"she got an A on her test\"",
        },
        {
            "example": "What best continues the sentence \"The sky was bright blue and so\"?",
            "continue": "\"it was a beautiful day\"",
        },
        {
            "example": "What best continues the sentence \"My mom cooked dinner because\"?",
            "continue": "\"we were hungry\"",
        },
        {
            "example": "What best continues the sentence \"My dad went to work early because\"?",
            "continue": "\"he had a lot to do\"",
        },
        {
            "example": "What best continues the sentence \"My sister was happy because\"?",
            "continue": "\"she got a new bike\"",
        }
    ])
]

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
            "statement": "Jonathan Samuels was born in the 70's.",
            "question": "Was Jonathan Samuels born in the 70's?"
        },
        {
            "statement": "Jerry bullied him and called him names",
            "question": "Did Jerry bully him and call him names?",
        },
        {
            "statement": "Sam and jade were going to go to the movies",
            "question": "Did did Sam and jade go to the movies?",
        },
        {
            "statement": "Chocolate is tasty, when I am feeling hungry.",
            "question": "Does chocolate taste good when you are hungry?",
        },
        {
            "statement": "Mark ran fast.",
            "question": "Did mark run fast?",
        },
        {
            "statement": "The sun rises in the east.",
            "question": "Does the sun rise in the east?",
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
            "statement": "The book was written by John.",
            "question": "Was the book written by John?",
        },
        {
            "statement": "I was born in the year 2000.",
            "question": "Were you born in the year 2000?",
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
            "statement": "The children went to the park to play.",
            "question": "Did the children go to the park to play?",
        },
        {
            "statement": "The cat ran away.",
            "question": "Did the cat run away?",
        },
        {
            "statement": "The dog barked at the mailman.",
            "question": "Did the dog bark at the mailman?",
        },
        {
            "statement": "The teacher gave the students a test.",
            "question": "Did the teacher give the students a test?",
        },
       {
            "statement": "The students studied for the test.",
            "question": "Did the students study for the test?",
        }
    ]),
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
            "statement": "The little girl was very excited.",
            "question": "Was the little girl excited?"
        },
      {
            "statement": "The teacher gave the students an assignment.",
            "question": "Did the teacher give the students an assignment?"
        },
      {
            "statement": "The sky was a beautiful shade of blue.",
            "question": "Was the sky blue?"
        },
      {
            "statement": "The students studied for the test.",
            "question": "Did the students study for the test?"
        },
       {
            "statement": "The mother cooked dinner for the family.",
            "question": "Did the mother cook dinner for the family?"
        }

    ])
]

openended_qa = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['context']}\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    required_keys=["passage", "question", "answer"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction=""
)

openended_qa_examples = [
    pd.DataFrame([
        {
            "context": "My aunt is a nurse and she often talks about long hours at work. Last week was especially bad and she was constantly working many hours.",
            "question": "Was her work easy?",
            "answer": "No, it was hard work."
        },
        {
            "context": "My roommate was sick. She stayed home from work and school. She slept all day long and by the end of the day, she was feeling better.",
            "question": "Did the rest help her?",
            "answer": "Yes, she slept and felt better."
        },
        {
            "context": "Andy had always wanted a big kids bike. When he turned six Year's old he asked for a bike for his birthday. He did not know how to ride a bike. On Andy's birthday his mother gave him a bike.",
            "question": "Did he cry all night?",
            "answer": "No, Andy was happy because he got a bike."
        },
        {
            "context": "John was a great basketball player. He would practice for hours and never get tired. He always tried his hardest and never gave up.",
            "question": "Was John good at basketball?",
            "answer": "Yes, John was a great basketball player."
        },
        {
            "context": "The dinner was a disaster. The food was overcooked and the guests were not happy. Everyone left early and the host was embarrassed.",
            "question": "Were the guests happy?",
            "answer": "No, the guests were not happy."
        },
        {
            "context": "The campground was very quiet. There were no other campers and the only sound was of the wind in the trees. It was a peaceful place to be.",
            "question": "Was it loud at the campground?",
            "answer": "No, it was very quiet."
        },
        {
            "context": "The car ride was long and boring. The driver kept talking and the passengers were getting tired. They stopped for a break and everyone was relieved.",
            "question": "Was the drive fun?",
            "answer": "No, the drive was long and boring."
        },
        {
            "context": "Kristin was a great student. She studied hard and was always prepared for her classes. She got good grades and was always praised by her teachers.",
            "question": "Was Kristin a good student?",
            "answer": "Yes, Kristin was a great student."
        },
        {
            "context": "My friend went on a camping trip. He was nervous because he had never been camping before. But by the end of the trip, he was having a great time.",
            "question": "Was he scared at first?",
            "answer": "Yes, he was nervous because he had never been camping before."
        },
        {
            "context": "The movie was a hit. People loved the story and the acting was great. Everyone was talking about it and it was a box office success.",
            "question": "Was the movie popular?",
            "answer": "Yes, the movie was a hit."
        }
    ]),
    pd.DataFrame([
        {
            "context": "It was a beautiful summer day outside. Bob decided to go for a walk at the park. Bob walked along the path and admired the scenery. He found a twenty dollar bill on the ground.",
            "question": "Was he disappointed?",
            "answer": "No, he was happy he got money."
        },
        {
            "context": "Mike is a busy man. He often eats fast food for breakfast. Mike wanted to enjoy a healthier breakfast. He tried an overnight oatmeal recipe.",
            "question": "Did Mike eat the oatmeal?",
            "answer": "Yes"
        },
        {
            "context": "Gina's sister cut her ankle on broken glass. The blood ran down her foot and into her shoe. When she saw the blood she ran home. Gina ran behind her, but couldn't keep up.",
            "question": "Did Gina's sister go to the doctor?",
            "answer": "Yes, because she was bleeding"
        },
        {
            "context": "John and Sarah went on a picnic. They had planned to go to the park, but it was raining heavily. So they decided to have their picnic on the rooftop. They set up their blankets and had a wonderful time.",
            "question": "Did they go to the park?",
            "answer": "No, they had their picnic on the rooftop because it was raining"
        },
        {
            "context": "Adam was working on his project late into the night. He was so tired that he couldn't keep his eyes open. He decided to take a nap in the office chair.",
            "question": "Was Adam able to finish his project?",
            "answer": "It is not known, because he took a nap before finishing it" 
        },
        {
            "context": "Bob was walking to the store when he saw a dog on the side of the road. He realized that the dog was injured and in need of help. He stopped and picked up the dog, taking it to the vet.",
            "question": "What did Bob do?",
            "answer": "He stopped and picked up the dog, taking it to the vet"
        },
        {
            "context": "Mary was walking to school when she saw a kitten stuck in a tree. She decided to climb up the tree to rescue the kitten. She was able to get the kitten down safely.",
            "question": "Did Mary rescue the kitten?",
            "answer": "Yes, she was able to get the kitten down safely"
        },
        {
            "context": "Dave was walking in the park when he saw a bird with a broken wing. He wanted to help the bird, so he carefully picked it up and took it to the nearest animal shelter.",
            "question": "What did Dave do?",
            "answer": "He carefully picked up the bird and took it to the nearest animal shelter"
        },
        {
            "context": "Sally was walking home from school when she heard a loud noise in the alley. She saw a group of kids playing with a ball and decided to join them. She had a great time playing with the kids.",
            "question": "What did Sally do?",
            "answer": "She saw a group of kids playing with a ball and decided to join them"
        },
        {
            "context": "John was walking to the store when he saw a homeless man sitting on the side of the road. He wanted to help, so he stopped and gave the man some money and food.",
            "question": "What did John do?",
            "answer": "He stopped and gave the homeless man some money and food"
        }
    ]),
    pd.DataFrame([
        {
            "context": "My aunt is a nurse she works a lot. Last week was especially bad and she was constantly working many hours.",
            "question": "Was her work easy?",
            "answer": "No"
        },
        {
            "context": "It was a beautiful day outside. Bob decided to go for a walkk. Bob walked along the path and admired the scenery. He found a twenty dollar bill on the ground.",
            "question": "Was he disappointed?",
            "answer": "No, he was happy he got money."
        },
        {
            "context": "Mom didn't want to cook dinner tonight. We were all very hungry. She told us to fend for ourselves. We ate cold cereal for dinner tonight.",
            "question": "Was everyone upset about the dinner?",
            "answer": "Yes, the food was cold"
        },
        {
            "context": "My friend was studying for a test. She spent all day studying and was feeling very tired. She went to bed early.",
            "question": "Did she feel prepared for the test?",
            "answer": "Yes, she felt prepared."
        },
        {
            "context": "My brother tried to fix our car but he couldn't figure out what was wrong. He called a mechanic who was able to fix it.",
            "question": "Was my brother frustrated?",
            "answer": "Yes, he was frustrated."
        },
        {
            "context": "My family went to the beach for a vacation. We had a great time swimming and playing in the sand. We also built a sandcastle.",
            "question": "Did my family have fun at the beach?",
            "answer": "Yes, they had a lot of fun."
        },
        {
            "context": "I was running late for work. I rushed to get ready but I still missed the bus. I had to take a cab to work.",
            "question": "Was I happy about taking a cab?",
            "answer": "No, I was not happy."
        },
        {
            "context": "We ordered pizza for dinner. Everyone was so excited to eat it. When we opened the box, the pizza was cold.",
            "question": "Were we disappointed?",
            "answer": "Yes, we were disappointed."
        },
        {
            "context": "My dad was feeling down. I tried to cheer him up and told him some jokes. He laughed and seemed to feel better.",
            "question": "Did my dad feel better after the jokes?",
            "answer": "Yes, he felt better."
        },
        {
            "context": "My friend just got a new job. She was very excited and looking forward to starting her new job. She was nervous but also excited.",
            "question": "Was my friend happy about her new job?",
            "answer": "Yes, she was happy."
        }
    ]),
]

sentiment = InputOutputPrompt(
    input_formatter=lambda x: f"Passage: {x['statement']}",
    output_formatter=lambda x: f"Sentiment: {x['sentiment']}",
    required_keys=["statement", "question"],
    input_output_sep="\n",
    example_sep="\n\n",
    instruction="Is the sentiment of the passage positive, negative, or neutral?\n\n"
)

sentiment_examples = [
    pd.DataFrame([
        {
            "statement": "Mary saw the animal",
            "sentiment": "neutral",
        },
        {
            "statement": "the town is quaint , but ultimately too boring and ugly",
            "sentiment": "negative",
        },
        {
            "statement": "he's a strong athlete, people come from miles away to watch him compete",
            "sentiment": "positive",
        },
        {
            "statement": "the food was mediocre at best",
            "sentiment": "negative",
        },
        {
            "statement": "the new movie was exciting and funny",
            "sentiment": "positive",
        },
        {
            "statement": "the professor was knowledgeable but unengaging",
            "sentiment": "neutral",
        },
        {
            "statement": "the weather was sunny and warm",
            "sentiment": "positive",
        },
        {
            "statement": "the meal was terrible and expensive",
            "sentiment": "negative",
        },
        {
            "statement": "the house was large and comfortable",
            "sentiment": "positive",
        },
        {
            "statement": "the coursework was difficult but manageable",
            "sentiment": "neutral",
        }
    ]),
    pd.DataFrame([
        {
            "statement": "Mary saw the animal",
            "sentiment": "neutral",
        },
        {
            "statement": "the town is quaint , but ultimately too boring and ugly",
            "sentiment": "negative",
        },
        {
            "statement": "he's a strong athlete, people come from miles away to watch him compete",
            "sentiment": "positive",
        },
        {
            "statement": "the new restaurant has excellent food and great service",
            "sentiment": "positive",
        },
        {
            "statement": "he was so mad he threw his shoe at the wall",
            "sentiment": "negative",
        },
        {
            "statement": "the new movie was entertaining, but not particularly special",
            "sentiment": "neutral",
        },
        {
            "statement": "she was so kind and helpful, I was truly grateful",
            "sentiment": "positive",
        },
        {
            "statement": "the food was terrible, I couldn't even finish it",
            "sentiment": "negative",
        },
        {
            "statement": "the weather was nice today, it was a great day for a walk",
            "sentiment": "positive",
        },
        {
            "statement": "the new policy was confusing and difficult to understand",
            "sentiment": "negative",
        }
    ]),
    pd.DataFrame([
        {
            "statement": "Mary saw the animal",
            "sentiment": "neutral",
        },
        {
            "statement": "the town is quaint , but ultimately too boring and ugly",
            "sentiment": "negative",
        },
        {
            "statement": "he's a strong athlete, people come from miles away to watch him compete",
            "sentiment": "positive",
        },
        {
            "statement": "the movie was terrible",
            "sentiment": "negative",
        },
        {
            "statement": "the food was delicious and plentiful",
            "sentiment": "positive",
        },
        {
            "statement": "the hotel room was comfortable and clean",
            "sentiment": "positive",
        },
        {
            "statement": "the play was well written but the acting was poor",
            "sentiment": "neutral",
        },
        {
            "statement": "she's a talented singer, her concerts are always sold out",
            "sentiment": "positive",
        },
        {
            "statement": "the new phone is slow and unreliable",
            "sentiment": "negative",
        },
        {
            "statement": "the restaurant had decent food but poor service",
            "sentiment": "neutral",
        }
    ])
]

what_next2 = InputOutputPrompt(
    input_formatter=lambda x: f"Choices:\n- {x['choice_a']}\n- {x['choice_b']}\n\nPassage: {x['passage']}",
    output_formatter=lambda x: f"{x['answer']}",
    required_keys=["choice_a", "choice_b", "passage", "answer"],
    input_output_sep=" ",
    example_sep="\n\n----\n\n",
    instruction="Pick the best choice for the passage.\n\n"
)

what_next_examples2 = [
    pd.DataFrame([
        {
            "passage": "My dad often talks about long hours at work. Because?",
            "choice_a": "work is hard",
            "choice_b": "work is easy",
            "answer": "work is hard"
        },
        {
            "passage": "My roommate was sick and took medicine. So?",
            "choice_a": "she threw a party",
            "choice_b": "she felt better",
            "answer": "she felt better"
        },
        {
            "passage": "Andy's parents got him a new toy. So?",
            "choice_a": "he played",
            "choice_b": "he cried",
            "answer": "he played"
        },
        {
            "passage": "The test was difficult. So?",
            "choice_a": "everyone passed",
            "choice_b": "everyone failed",
            "answer": "everyone failed"
        },
        {
            "passage": "The teacher gave the students a break. So?",
            "choice_a": "they went to lunch",
            "choice_b": "they stayed in class",
            "answer": "they went to lunch"
        },
        {
            "passage": "It started to rain. So?",
            "choice_a": "people stayed inside",
            "choice_b": "people went outside",
            "answer": "people stayed inside"
        },
        {
            "passage": "The football team won the game. So?",
            "choice_a": "they were disappointed",
            "choice_b": "they were excited",
            "answer": "they were excited"
        },
        {
            "passage": "The dentist said to brush twice a day. So?",
            "choice_a": "you don't need to floss",
            "choice_b": "you should floss too",
            "answer": "you should floss too"
        },
        {
            "passage": "The bell rang. So?",
            "choice_a": "class started",
            "choice_b": "class ended",
            "answer": "class ended"
        },
        {
            "passage": "The alarm went off. So?",
            "choice_a": "it was time to sleep",
            "choice_b": "it was time to wake up",
            "answer": "it was time to wake up"
        }
    ]),
    pd.DataFrame([
        {
            "passage": "The girl went to college and graduated with honors.",
            "choice_a": "She was qualified to get a job.",
            "choice_b": "She was qualified to eat pizza.",
            "answer": "she was qualified to get a job."
        },
        {
            "passage": "Max bought all his friends cupcakes for the party.",
            "choice_a": "They never spoke to him again.",
            "choice_b": "They all thanked him.",
            "answer": "They all thanked him."
        },
        {
            "passage": "Sam felt so hungry so he bought himself some cheese!",
            "choice_a": "After he ate the cheese, he was starving.",
            "choice_b": "After he ate the cheese, he felt better.",
            "answer": "After he ate the cheese, he felt better."
        },
        {
            "passage": "The dog was not feeling well so the owner took him to the vet.",
            "choice_a": "The vet said he would be okay.",
            "choice_b": "The vet said he needed a bath.",
            "answer": "The vet said he would be okay."
        },
        {
            "passage": "The runner crossed the finish line with a time of 30 minutes.",
            "choice_a": "He was the fastest runner.",
            "choice_b": "He was the slowest runner.",
            "answer": "He was the fastest runner."
        },
        {
            "passage": "The student had his work graded and received an A.",
            "choice_a": "He was proud of his grade.",
            "choice_b": "He was angry about his grade.",
            "answer": "He was proud of his grade."
        },
        {
            "passage": "The teacher gave the class a quiz.",
            "choice_a": "The class was excited.",
            "choice_b": "The class was sad.",
            "answer": "The class was excited."
        },
        {
            "passage": "The family went to the beach on vacation.",
            "choice_a": "They all had a great time.",
            "choice_b": "They all had a terrible time.",
            "answer": "They all had a great time."
        },
        {
            "passage": "The musician sang a song for the audience.",
            "choice_a": "The audience was impressed.",
            "choice_b": "The audience was bored.",
            "answer": "The audience was impressed."
        },
        {
            "passage": "The painter completed the painting.",
            "choice_a": "He was happy with the results.",
            "choice_b": "He was disappointed with the results.",
            "answer": "He was happy with the results."
        },
        {
            "passage": "The student studied for the exam.",
            "choice_a": "He passed the exam.",
            "choice_b": "He failed the exam.",
            "answer": "He passed the exam."
        }
    ]),
    pd.DataFrame([
      {
            "passage": "Sam and Jade were excited to see the new movie.",
            "choice_a": "They went to the theater.",
            "choice_b": "They went swimming.",
            "answer": "They went to the theater."
        },
        {
            "passage": "Matt is very competitive in soccer.",
            "choice_a": "He practices all the time.",
            "choice_b": "He loves to lose.",
            "answer": "He practices all the time."
        },
        {
            "passage": "She can read the entire book in a single day.",
            "choice_a": "She is a slow reader.",
            "choice_b": "She is a fast reader.",
            "answer": "She is a fast reader."
        },
        {
            "passage": "The house was built by a master carpenter.",
            "choice_a": "The house is very modern.",
            "choice_b": "The house is very well-crafted.",
            "answer": "The house is very well-crafted."
        },
        {
            "passage": "The puppy is always so energetic.",
            "choice_a": "The puppy loves to play.",
            "choice_b": "The puppy loves to sleep.",
            "answer": "The puppy loves to play."
        },
        {
            "passage": "The class is studying mathematics.",
            "choice_a": "They are learning about science.",
            "choice_b": "They are learning about numbers.",
            "answer": "They are learning about numbers."
        },
        {
            "passage": "He was born with a gift for music.",
            "choice_a": "He loves to sing.",
            "choice_b": "He loves to dance.",
            "answer": "He loves to sing."
        },
        {
            "passage": "The garden was filled with colorful flowers.",
            "choice_a": "The garden was very dull.",
            "choice_b": "The garden was very vibrant.",
            "answer": "The garden was very vibrant."
        },
        {
            "passage": "She was a talented artist.",
            "choice_a": "She loved to paint.",
            "choice_b": "She loved to draw.",
            "answer": "She loved to paint."
        },
        {
            "passage": "The family was enjoying a picnic at the park.",
            "choice_a": "They were playing games.",
            "choice_b": "They were eating lunch.",
            "answer": "They were eating lunch."
        }
    ])
]


class COPADecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def zero_few_baseline(
        self,
        test_data,
        few_shot_df,
        manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer,
        do_few_shot=True,
    ):
        expt_log = {}
        total = 0
        total_crct = 0 

        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            if ind in expt_log:
                pred = entry["pred"]
                gold = entry["gold"]
            else:
                icl_str = ""

                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        s_text = s_row['inputs_pretokenized'].replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                        s_parts = s_text.split(". ")
                        s_sentence = s_parts[0]
                        s_transition = s_parts[1]
                        options = [l for l in s_text.split("\n") if l.startswith("- ")]
                        if "as a consequence" in s_transition:
                            s_text = f"{s_sentence} so"
                        elif "as a result of" in s_transition:
                            s_text = f"{s_sentence} because"
                        icl_str += f"Context: {s_text} {s_row['targets_pretokenized']}\n\n"
                
                text = row['inputs_pretokenized']
                parts = text.split(". ")
                sentence = parts[0]
                transition = parts[1]
                options = [l for l in text.split("\n") if l.startswith("- ")]
                if "as a consequence" in transition:
                    text = f"{sentence} so"
                elif "as a result of" in transition:
                    text = f"{sentence} because"
                text = text.replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                gold = row['targets_pretokenized']
                prompt = f"Pick the more likely continuation to the following sentence.\n\n\n{icl_str}Context: {{text:}}"
                pmp = prompt.format(text=text)
                if i == 0:
                    print(pmp)
                raw_answer, _ = get_response(
                    pmp,
                    manifest_answer,
                    gold_choices=[options[0].replace("- ", "").strip(), options[1].replace("- ", "").strip()],
                    overwrite=bool(overwrite_manifest_answer),
                    max_toks=50,
                )
                answer = raw_answer.strip().lower()
                answer = answer.split("\n")
                answer = [a for a in answer if a]
                if answer:
                    answer = answer[0].replace("-", "").strip()
                else:
                    answer = ''

                pred = "".join([a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']])
                gold = "".join([a for a in gold if a not in [".", ",", "?", ";", ":", "'", '"']])
                
                crct = gold.lower() == pred.lower()
                total += 1
                total_crct += crct

                entry = {
                    "ind": ind,
                    "example": text,
                    "base_prompt": pmp,
                    "raw_answer": raw_answer,
                    "pred": pred,
                    "gold": gold,
                }
                expt_log[ind] = entry

        accuracy = total_crct/total
        return expt_log, accuracy

    def get_boost_decomp_examples(self, train_data, boost_id):
        if boost_id < 1: 
            return [
                what_next_examples[boost_id],
            ]
        elif boost_id < 2:
            return [
                what_next_examples2[boost_id-1],
            ]
        elif boost_id >= 2:
            seed = [1, 2, 3][boost_id-2]
            k_shot = 4*seed
            random.seed(seed)
            np.random.seed(seed)

            data_train = pd.DataFrame(train_data)
            sub_df = data_train.sample(k_shot)
            booster_df = sub_df.sample(frac=1, random_state=0)
            print(f"Selected: {len(booster_df)} in context examples.")
            return [
                booster_df
            ]

    def what_happened_next(self, prompt, boost_ex, example, transition, choice_a, choice_b, word, manifest, overwrite_manifest):
        example = example.strip(".")
        choice_a = choice_a.lower()
        choice_b = choice_b.lower()
        transition = transition.strip()
        prompt_suffix = prompt(boost_ex)
        ex_prompt = f"{prompt_suffix}\n\nQuestion: ({{word:}} \'{{choice_a:}}\', {{word:}} \'{{choice_b:}}\') {{example:}} {{word:}}?"
        raw_answer = get_response(
            ex_prompt.format(word=word, choice_a=choice_a, choice_b=choice_b, example=example), 
            manifest,
            max_toks= 4*len(choice_a.split()),
            overwrite=bool(overwrite_manifest))
        
        # print(f"raw_answer: ", raw_answer)
        try:
            answer = [q for q in raw_answer.split("\n") if q][0].lower()
        #######################
        except:
            answer = ''
        #######################            
            
        pred = ''
        for n in range(5,50):
            for idx_offset in range(len(answer) - n + 1):
                ngram = "".join(answer[idx_offset:idx_offset+n])
                if ngram in choice_a and ngram not in choice_b:
                    pred = choice_a
                elif ngram not in choice_a and ngram in choice_b:
                    pred = choice_b
        return pred, ex_prompt

    def question_answer(self, prompt, boost_ex, example, transition, choice_a, choice_b, word, manifest, overwrite_manifest):
        example = example.strip(".")
        choice_a = choice_a.lower()
        choice_b = choice_b.lower()
        transition = transition.strip()
        prompt_suffix = prompt(boost_ex)
        ex_prompt = f"{prompt_suffix}\n\nQuestion: What best continues the sentence \"{{example:}}\"?\nAnswer:"
        ex_pmp = ex_prompt.format(example=example)
        raw_answer, log_prob = get_response(
            ex_pmp, 
            manifest,
            gold_choices=[choice_a, choice_b],
            max_toks= 4*len(choice_a.split()),
            overwrite=bool(overwrite_manifest))
        answer = [q for q in raw_answer.split("\n") if q][0].lower()
        pred = ''
        for n in range(5,50):
            for idx_offset in range(len(answer) - n + 1):
                ngram = "".join(answer[idx_offset:idx_offset+n])
                if ngram in choice_a and ngram not in choice_b:
                    pred = '1'
                elif ngram not in choice_a and ngram in choice_b:
                    pred = '2'
        if not pred:
            import pdb;
            pdb.set_trace()
        return pred, ex_pmp

    def answer_question(self, question, passage, all_prompts, boost_examples, manifest, overwrite_manifest, option=1):
        one_at_a_time = all_prompts[1](boost_examples[1])

        answer_prompt = f"{one_at_a_time}\n\nPassage: {{passage:}}\nQuestion: {{question:}}\n"
        answer = get_response(
            answer_prompt.format(passage=passage, question=question), 
            manifest, 
            max_toks=50)
        answer = answer.replace("Answer: ", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0].replace(",", "").replace(".", "").lower()
        else:
            answer = ''
        pred = ''
        if option == 1:
            if 'yes' in answer.split():
                pred = "1"
            elif 'no' in answer.split():
                pred = "2"
        elif option == 2:
            if 'no' in answer.split():
                pred = "1"
            elif 'yes' in answer.split():
                pred = "2"
        return pred, answer_prompt

    def get_one_by_one(self, example, choice_a, choice_b, all_prompts, boost_examples, manifest, overwrite_manifest):

        # construct questions
        question_a, questioner_prompt = self.get_question(choice_a, all_prompts, boost_examples, manifest, overwrite_manifest)
        question_b, questioner_prompt = self.get_question(choice_b, all_prompts, boost_examples, manifest, overwrite_manifest)

        # ask questions
        pred_a, answerer_prompt = self.answer_question(question_a, example, all_prompts, boost_examples, manifest, overwrite_manifest, option=1)
        pred_b, answerer_prompt = self.answer_question(question_b, example, all_prompts, boost_examples, manifest, overwrite_manifest, option=2)
        
        # reconcile answer
        if pred_a == "1" and pred_b == "1":
            pred = choice_a
        elif pred_a == "2" and pred_b == "2":
            pred = choice_b
        elif pred_a and not pred_b:
            if pred_a == "1":
                pred = choice_a
            else:
                pred = choice_b
        elif not pred_b and pred_b:
            if pred_b == "1":
                pred = choice_a
            else:
                pred = choice_b
        else:
            pred = ''
        return pred, questioner_prompt, answerer_prompt

    def get_sentiment(self, statement, all_prompts, boost_examples, manifest, overwrite_manifest):
        sentiment_prompt = all_prompts[0](boost_examples[0])
        prompt = f"{sentiment_prompt}\n\nPassage: {{statement:}}\nSentiment: "
        raw_answer = get_response(
            prompt.format(statement=statement), 
            manifest,
            max_toks=5)
        sent = raw_answer.split("\n")[0]

        if "positive" in sent:
            sent = 1
        elif "negative" in sent:
            sent = -1
        elif "neutral" in sent:
            sent = 0

        return sent, sentiment_prompt

    def combine_sentiments(self, example, choice_a, choice_b, all_prompts, boost_examples, manifest, overwrite_manifest):

        # construct questions
        sentiment_a, sentiment_prompt = self.get_sentiment(choice_a, all_prompts, boost_examples, manifest, overwrite_manifest)
        sentiment_b, sentiment_prompt = self.get_sentiment(choice_b, all_prompts, boost_examples, manifest, overwrite_manifest)
        sentiment_ex, sentiment_prompt = self.get_sentiment(example, all_prompts, boost_examples, manifest, overwrite_manifest)
        
        # reconcile answer
        pred = ''
        if abs(sentiment_a - sentiment_ex) < abs(sentiment_b - sentiment_ex):
            pred = choice_a
        elif abs(sentiment_a - sentiment_ex) > abs(sentiment_b - sentiment_ex):
            pred = choice_b
        return pred, sentiment_prompt

    def get_question(self, statement, all_prompts, boost_examples, manifest, overwrite_manifest):
        questioner = all_prompts[0](boost_examples[0])

        question_prompt = f"{questioner}\n\nStatement: {{statement:}}\n"
        question = get_response(
            question_prompt.format(statement=statement), 
            manifest, 
            max_toks= 4*len(statement.split()))
        question = question.replace("Question: ", "")
        question = [q for q in question.split("\n") if q]
        if not question:
            question = f"{statement} Yes or no?"
        else:
            question = question[0]
        return question, question_prompt


    def get_what_next(self, example, choice_a, choice_b, transition, all_prompts, boost_examples, manifest, overwrite_manifest):
        what_next_prompt = all_prompts[0](boost_examples[0])
        if "result of":
            prompt = f"{what_next_prompt}\n\n----\n\nChoices:\n- {{choice_a:}}\n- {{choice_b:}}\n\nPassage: {{example:}} Because?"
        elif "consequence":
            prompt = f"{what_next_prompt}\n\n----\n\nChoices:\n- {{choice_a:}}\n- {{choice_b:}}\n\nPassage: {{example:}} So?"
        raw_answer = get_response(
            prompt.format(choice_a=choice_a, choice_b=choice_b, example=example), 
            manifest,
            max_toks=50)
        answer = raw_answer.split("\n")[0].lower()
        choice_a = choice_a.lower()
        choice_b = choice_b.lower()
        pred = ''
        for n in range(5,50):
            for idx_offset in range(len(answer) - n + 1):
                ngram = "".join(answer[idx_offset:idx_offset+n])
                if ngram in choice_a and ngram not in choice_b:
                    pred = choice_a
                elif ngram not in choice_a and ngram in choice_b:
                    pred = choice_b
        return pred, what_next_prompt


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
            individual_accuracies.append(classification_report(labels, [p[i] for p in all_boost_preds], output_dict=True)["accuracy"])
        report = classification_report(labels, preds, output_dict=True)
        return expt_log, expt_log_train, report["accuracy"], individual_accuracies

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
            text = row['inputs_pretokenized']
            text = text.replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
            gold = row['targets_pretokenized']
            parts = text.split("\n")
            statement = parts[0].split(".")[0:-1]
            transition = parts[0].split(".")[-1]
            example = " ".join(statement)
            choice_a = parts[1].replace("-", "").strip()
            choice_b = parts[2].replace("-", "").strip()
            gold_idx = -1
            if gold.lower() == choice_a.lower():
                gold_idx = '1'
            else:
                gold_idx = '2'

            all_prompts = []

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_num, boost_examples in enumerate(boost_dfs):
                icl_str = ""
                pred = ''
                answer2 = None

                if boost_num < 1:
                    all_prompts = []
                    if 'as a consequence' in transition:
                        answer, what_next_prompt = self.what_happened_next(question, boost_examples[0], example, transition, choice_a, choice_b, 'and so', manifest_answer, overwrite_manifest_answer)
                    else:
                        answer, what_next_prompt = self.what_happened_next(
                            question, boost_examples[0], example, transition, choice_a, choice_b, 'because', manifest_answer, overwrite_manifest_answer)

                    if 'as a consequence' in transition:
                        answer2, what_next_prompt = self.what_happened_next(question, boost_examples[0], example, transition, choice_b, choice_a, 'and so', manifest_answer, overwrite_manifest_answer)
                    else:
                        answer2, what_next_prompt = self.what_happened_next(
                            question, boost_examples[0], example, transition, choice_b, choice_a, 'because', manifest_answer, overwrite_manifest_answer)

                    if answer != answer2:
                        answer = ''

                    all_prompts.append(what_next_prompt)


                elif boost_num < 2:
                    answer, what_next_prompt = self.get_what_next(
                        example, choice_a, choice_b, transition, [what_next2], boost_examples, manifest_answer, overwrite_manifest_answer
                    )
                    answer2, what_next_prompt = self.get_what_next(
                        example, choice_b, choice_a, transition, [what_next2], boost_examples, manifest_answer, overwrite_manifest_answer
                    )
                    if answer != answer2:
                        answer = ''

                    all_prompts.append(what_next_prompt)
                    
                else:
                    icl_str = ""
                    for s_ind, s_row in boost_examples[0].iterrows():
                        s_text = s_row['inputs_pretokenized'].replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                        s_parts = s_text.split(". ")
                        s_sentence = s_parts[0]
                        s_transition = s_parts[1]
                        options = [l for l in s_text.split("\n") if l.startswith("- ")]
                        if "as a consequence" in s_transition:
                            s_text = f"{s_sentence} so"
                        elif "as a result of" in s_transition:
                            s_text = f"{s_sentence} because"
                        s_gold = s_row['targets_pretokenized'].lower()
                        icl_str += f"Context: {s_text} {s_gold}\n\n"
                    
                    text = row['inputs_pretokenized']
                    parts = text.split(". ")
                    sentence = parts[0]
                    transition = parts[1]
                    options = [l.lower() for l in text.split("\n") if l.startswith("- ")]
                    if "as a consequence" in transition:
                        text = f"{sentence} so"
                    elif "as a result of" in transition:
                        text = f"{sentence} because"
                    text = text.replace("Pick the more likely continuation to the following sentence:", "").strip("\n")
                    gold = row['targets_pretokenized']
                    prompt = f"Pick the more likely continuation to the following sentence.\n\n\n{icl_str}Context: {{text:}}"
                    if i == 0:
                        print(prompt.format(text=text))
                    all_prompts.append(prompt)
                    raw_answer, _ = get_response(
                        prompt.format(text=text),
                        manifest_answer,
                        gold_choices=[options[0].replace("- ", "").strip(), options[1].replace("- ", "").strip()],
                        overwrite=bool(overwrite_manifest_answer),
                        max_toks=50,
                    )
                    answer = raw_answer.strip().lower()
                    answer = answer.split("\n")
                    answer = [a for a in answer if a]
                    if answer:
                        answer = answer[0].replace("-", "").strip()
                    else:
                        answer = ''

                pred = "".join([a for a in answer if a not in [".", ",", "?", ";", ":", "'", '"']]).lower()
                gold = "".join([a for a in gold if a not in [".", ",", "?", ";", ":", "'", '"']]).lower()

                prompts_across_boost.append(all_prompts)
                preds_across_boost.append(pred)

            
            preds_across_boost.reverse()
            mapped_p = []
            for p in preds_across_boost:
                if not p:
                    mapped_p.append("")
                    continue
                if p == gold:
                    mapped_p.append(gold_idx)
                elif gold_idx == "1":
                    mapped_p.append("2")
                else:
                    mapped_p.append("1")

            all_boost_preds.append(mapped_p)

            entry = {
                "ind": ind,
                "example": text,
                "prompts": prompts_across_boost,
                "preds_boost": mapped_p,
                "gold": gold_idx,
            }
            expt_log[ind] = entry
            
            labels.append(gold_idx)
        return expt_log, all_boost_preds, labels


def main():
    args = get_args()
    args.num_boost = 5
    task_name = "super_glue_copa"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_copa_more_likely/"
    decomp = COPADecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
