#!/usr/bin/env python
# coding: utf-8
from tqdm.auto import tqdm
import pandas as pd
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

from sklearn.metrics import classification_report
from decomposition import Decomposition, get_args, DATA_DIR
from utils import get_response, InputOutputPrompt

##############################################################################################################################

# ################
# import os
# import json
# model_name_question = os.environ['EXP_MODE_QUESTION']
# # question_file = '/nvmedata/changranh/ama_question_synthetic_data/' + model_name_question + self.task_name + ".jsonl"
# question_file = '/scratch/changranh/ama_question_synthetic_data/' + model_name_question + '_RTE' + ".jsonl"        
# ################  


questioner = InputOutputPrompt(
    input_formatter=lambda x: f"Statement: {x['statement']}",
    output_formatter=lambda x: f"Question: {x['question']}",
    input_output_sep="\n",
    example_sep="\n\n",
    required_keys=["question", "statement"],
    instruction="Rewrite the statement as a question.\n\n"
)

questioner_examples = [
    pd.DataFrame([
        {
            "statement": "most of the light comes from the sun",
            "question": "Does most of the light come from the sun?"
        },
        {
            "statement": "the test was not hard",
            "question": "Was the test hard?"
        },
        {
            "statement": "it was a good idea to buy your parents gifts",
            "question": "Was it a good idea to buy your parents gifts?"
        },
        {
            "statement": "The 20 cans will arrive in the grocery store tomorrow.",
            "question": "Will the 20 cans arrive in the grocery store tomorrow?"
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
            "statement": "the wind was blowing very fast",
            "question": "Was the wind blowing very fast?"
        },
        {
            "statement": "The bird couldn't fly away.",
            "question": "Could the bird fly away?"
        },
        {
            "statement": "The students were late to the class.",
            "question": "Were the students late to the class?"
        },
        {
            "statement": "She was the last one to leave the room.",
            "question": "Was she the last one to leave the room?"
        },
        {
            "statement": "The dog was barking all night.",
            "question": "Was the dog barking all night?"
        },
        {
            "statement": "The car was very expensive.",
            "question": "Was the car very expensive?"
        },
        {
            "statement": "He was the fastest runner in the race.",
            "question": "Was he the fastest runner in the race?"
        },
        {
            "statement": "They ate pizza for dinner.",
            "question": "Did they eat pizza for dinner?"
        },
        {
            "statement": "The cat was sleeping on the couch.",
            "question": "Was the cat sleeping on the couch?"
        },
        {
            "statement": "The mountain was beautiful.",
            "question": "Was the mountain beautiful?"
        },
        {
            "statement": "The river was too deep to cross.",
            "question": "Was the river too deep to cross?"
        },
        {
            "statement": "She was the leader of the team.",
            "question": "Was she the leader of the team?"
        },
        {
            "statement": "The movie was very funny.",
            "question": "Was the movie very funny?"
        },
        {
            "statement": "The ice cream was too cold.",
            "question": "Was the ice cream too cold?"
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
            "statement": "it was a good idea to buy your parents gifts",
            "question": "Was it a good idea to buy your parents gifts?"
        },
        {
            "statement": "The 20 cans will arrive in the grocery store tomorrow.",
            "question": "Will the 20 cans arrive in the grocery store tomorrow?"
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
            "statement": "The cake was delicious",
            "question": "Was the cake delicious?"
        },
        {
            "statement": "The new movie starts tomorrow",
            "question": "When does the new movie start?"
        },
        {
            "statement": "It was cloudy outside",
            "question": "Was it cloudy outside?"
        },
        {
            "statement": "The class went on a field trip yesterday.",
            "question": "When did the class go on a field trip?"
        },
        {
            "statement": "the weather was nice",
            "question": "Was the weather nice?"
        },
        {
            "statement": "The students worked on their project",
            "question": "Did the students work on their project?"
        },
        {
            "statement": "The teacher gave out extra credit",
            "question": "Did the teacher give out extra credit?"
        },
        {
            "statement": "The concert was sold out",
            "question": "Was the concert sold out?"
        },
        {
            "statement": "The family went to the beach",
            "question": "Where did the family go?"
        },
        {
            "statement": "the cat meowed",
            "question": "Did the cat meow?"
        },
        {
            "statement": "The store closed at 8 pm",
            "question": "When does the store close?"
        },
        {
            "statement": "the hockey game was cancelled",
            "question": "Was the hockey game cancelled?"
        },
        {
            "statement": "The house was painted yellow",
            "question": "What color was the house painted?"
        },
        {
            "statement": "The game was postponed",
            "question": "Was the game postponed?"
        }
    ]),

    pd.DataFrame([
        {
            "statement": "tennis can be played on grass courts",
            "question": "Can tennis be played on grass courts?",
        },
        {
            "statement": "the artist painted a picture of the apple in a bowl.",
            "question": "Did the artist paint a picture of an apple in a bowl?",
        },
        {
            "statement": "mary is unhappy with tim.",
            "question": "Is mary unhappy with Tim?",
        },
        {
            "statement": "after school, Jim was going to go to the park",
            "question": "Was Jim going to go to the park after school?",
        },
        {
            "statement": "the car was parked in the driveway.",
            "question": "Where was the car parked?",
        },
        {
            "statement": "there were four people in the room.",
            "question": "How many people were in the room?",
        },
        {
            "statement": "the sun was shining brightly in the sky.",
            "question": "Was the sun shining brightly in the sky?",
        },
        {
            "statement": "the cat chased the bird away.",
            "question": "What did the cat do to the bird?",
        },
        {
            "statement": "the manager gave the employees a day off.",
            "question": "Did the manager give the employees a day off?",
        },
        {
            "statement": "the man walked to the store.",
            "question": "Where did the man go?",
        },
        {
            "statement": "the elephant was the biggest animal in the zoo.",
            "question": "What was the biggest animal in the zoo?",
        },
        {
            "statement": "the teacher asked the students a difficult question.",
            "question": "Did the teacher ask the students a difficult question?",
        },
        {
            "statement": "the rain was pouring down.",
            "question": "What was happening with the rain?",
        },
        {
            "statement": "the party was held in the garden.",
            "question": "Where was the party held?",
        },
        {
            "statement": "jane was reading a book in her room.",
            "question": "What was Jane doing in her room?",
        },
        {
            "statement": "the children were playing in the playground.",
            "question": "Where were the children playing?",
        },
        {
            "statement": "the dog barked at the stranger.",
            "question": "What did the dog do to the stranger?",
        },
        {
            "statement": "the man was running in the park.",
            "question": "Where was the man running?",
        },
        {
            "statement": "the teacher was talking about history.",
            "question": "What was the teacher talking about?",
        },
        {
            "statement": "the sky was a beautiful shade of blue.",
            "question": "What color was the sky?",
        }
    ]),

    pd.DataFrame([
        {
            "statement": "she prefers kittens over puppies",
            "question": "What does she prefer over puppies?\nAnswer: kittens",
        },
        {
            "statement": "Max and his wife went on a trip to Europe",
            "question": "Where did Max and his wife go on a trip?\nAnswer: Europe",
        },
        {
            "statement": "jared was born during the war in 1942",
            "question": "Jared was born during a war in which year?\nAnswer: 1942",
        },
        {
            "statement": "it took jenna 7 attempts to solve the problem",
            "question": "How many attempts did it take Jenna to solve the problem?\nAnswer: 7",
        },
        {
            "statement": "the mountain is located in the western part of the state",
            "question": "Where is the mountain located?\nAnswer: western part of the state",
        },
        {
            "statement": "the store is open 7 days a week",
            "question": "How many days a week is the store open?\nAnswer: 7",
        },
        {
            "statement": "she won the race with a time of 8 minutes and 12 seconds",
            "question": "What was her time in the race?\nAnswer: 8 minutes and 12 seconds",
        },
        {
            "statement": "the team worked together for 6 months",
            "question": "How long did the team work together?\nAnswer: 6 months",
        },
        {
            "statement": "The restaurant is located on Main Street",
            "question": "Where is the restaurant located?\nAnswer: Main Street",
        },
        {
            "statement": "The show starts at 8 o'clock",
            "question": "What time does the show start?\nAnswer: 8 o'clock",
        },
        {
            "statement": "He graduated from college in 2003",
            "question": "What year did he graduate from college?\nAnswer: 2003",
        },
        {
            "statement": "The store has been open for 10 years",
            "question": "How long has the store been open?\nAnswer: 10 years",
        },
        {
            "statement": "The conference is taking place in Paris",
            "question": "Where is the conference taking place?\nAnswer: Paris",
        },
        {
            "statement": "The party starts at 7 pm",
            "question": "What time does the party start?\nAnswer: 7 pm",
        },
        {
            "statement": "The movie was released in 2017",
            "question": "In what year was the movie released?\nAnswer: 2017",
        },
        {
            "statement": "The store closes at 9 pm",
            "question": "What time does the store close?\nAnswer: 9 pm",
        },
        {
            "statement": "The store is located on First Avenue",
            "question": "Where is the store located?\nAnswer: First Avenue",
        },
        {
            "statement": "The class starts at 10 am",
            "question": "What time does the class start?\nAnswer: 10 am",
        },
        {
            "statement": "The book was published in 2010",
            "question": "What year was the book published?\nAnswer: 2010",
        },
        {
            "statement": "The race ended in a tie",
            "question": "How did the race end?\nAnswer: tie",
        }
    ]),
]

openended_qa = InputOutputPrompt(
    input_formatter=lambda x: f"Context: {x['passage']}\n\nQuestion: {x['question']}",
    output_formatter=lambda x: f"Answer: {x['answer']}",
    input_output_sep="\n\n",
    example_sep="\n\n----\n\n",
    required_keys=["question", "statement", 'answer'],
    instruction="Answer the question. If there is no evidence in the context, return \"Unknown\".\n\n"
)

openended_qa_examples = [
    pd.DataFrame([
        {
            "passage": "Jenna's 10th birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "Did 10 friends attend Jenna's party?",
            "answer": "Unknown, at least 10"
        },
        {
            "passage": "The bullies attacked John when he was walking through the elementary school parking lot and then got sent to the teacher's office.",
            "question": "Did the bullies attack John in the teacher's office?",
            "answer": "No, parking lot"
        },
        {
            "passage": "WISS discovered a new monkey disease in a remote tribe in the Amazon rainforrest last week. It was highly contagious.",
            "question": "Did WISS discover a new disease?",
            "answer": "Yes, new monkey disease"
        },
        {
            "passage": "The professor gave the students a difficult exam and asked them to turn it in by the end of the week.",
            "question": "When was the exam due?",
            "answer": "End of the week"
        },
        {
            "passage": "The Smith family drove to their vacation home in Florida last weekend and stayed for a week.",
            "question": "How long did the Smiths stay in Florida?",
            "answer": "A week"
        },
        {
            "passage": "The farmer harvested his crops and sold them at the local farmer's market this Saturday.",
            "question": "Where did the farmer sell his crops?",
            "answer": "Local farmer's market"
        },
        {
            "passage": "The students went to the school library to study for the final exam.",
            "question": "Where did the students go to study?",
            "answer": "School library"
        },
        {
            "passage": "The children ate their dinner at the kitchen table and then went outside to play.",
            "question": "Where did the children eat their dinner?",
            "answer": "Kitchen table"
        },
        {
            "passage": "The hikers took a break at the top of the mountain and enjoyed the view.",
            "question": "Where did the hikers take a break?",
            "answer": "Top of the mountain"
        },
        {
            "passage": "The firefighters rushed to the burning building and extinguished the blaze.",
            "question": "What did the firefighters do?",
            "answer": "Extinguished the blaze"
        },
        {
            "passage": "The football team won their championship game last night.",
            "question": "What did the football team do?",
            "answer": "Won their championship game"
        },
        {
            "passage": "The family went to the beach and had a picnic lunch.",
            "question": "Where did the family go?",
            "answer": "Beach"
        },
        {
            "passage": "The police arrested the suspect for the crime.",
            "question": "What did the police do?",
            "answer": "Arrested the suspect"
        },
        {
            "passage": "The detectives found the evidence they needed to solve the case.",
            "question": "What did the detectives do?",
            "answer": "Found the evidence"
        },
        {
            "passage": "The doctor prescribed the patient medication to help with her pain.",
            "question": "What did the doctor do?",
            "answer": "Prescribed medication"
        },
        {
            "passage": "The teacher gave the students an assignment to complete by the end of the day.",
            "question": "When was the assignment due?",
            "answer": "End of the day"
        },
        {
            "passage": "The chef cooked dinner for the family and served it to them at the dinner table.",
            "question": "Where did the chef serve dinner?",
            "answer": "Dinner table"
        },
        {
            "passage": "The construction workers built a new bridge over the river this month.",
            "question": "What did the construction workers do?",
            "answer": "Built a new bridge"
        },
    ]),

    pd.DataFrame([
        {
            "passage": "Jenna's birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "Did 10 friends attend Jenna's party?",
            "answer": "unknown, at least 10"
        },
        {
            "passage": "The bullies punched John when he was walking through the elementary school parking lot. They punched 3 times.",
            "question": "Did the bullies punch John 4 time?",
            "answer": "No, 3 times"
        },
        {
            "passage": "WISS discovered a new monkey disease in a remote tribe in the Amazon rainforrest last week. It was highly contagious.",
            "question": "Did WISS discover a new species of monkeys?",
            "answer": "Unknown"
        },
        {
            "passage": "Jenny was walking to school when she saw a large cat cross the street. It had black and white fur.",
            "question": "What color was the cat Jenny saw?",
            "answer": "Black and white"
        },
        {
            "passage": "The parents took their children to the beach last weekend. They stayed for 4 days.",
            "question": "How long did the family stay at the beach?",
            "answer": "4 days"
        },
        {
            "passage": "Jane and her family went to the zoo last Sunday. They saw at least 4 different species of animals.",
            "question": "How many different species of animals did Jane's family see?",
            "answer": "At least 4"
        },
        {
            "passage": "The cat knocked over the vase and broke it. It had been in the family for 5 generations.",
            "question": "How long had the vase been in the family before it was broken?",
            "answer": "5 generations"
        },
        {
            "passage": "The mayor visited the school last week. He brought a gift of books with him.",
            "question": "What did the mayor bring to the school?",
            "answer": "Books"
        },
        {
            "passage": "The teacher gave the students a quiz today. It had 10 questions.",
            "question": "How many questions were on the quiz?",
            "answer": "10"
        },
        {
            "passage": "Bob and his family took a trip to the mountains this summer. They were there for 7 days.",
            "question": "How long did Bob's family stay in the mountains?",
            "answer": "7 days"
        },
        {
            "passage": "The park ranger found a nest of baby birds in an old tree. He counted at least 10 baby birds in the nest.",
            "question": "How many baby birds were in the nest?",
            "answer": "At least 10"
        },
        {
            "passage": "The students were studying in the library all day. They had 8 books checked out.",
            "question": "How many books did the students have checked out?",
            "answer": "8"
        },
        {
            "passage": "The family had a picnic in the park. They stayed for 3 hours.",
            "question": "How long did the family stay at the park for the picnic?",
            "answer": "3 hours"
        },
        {
            "passage": "The hikers saw 8 different types of wild animals on their adventure. 5 of them were never seen before.",
            "question": "How many kinds of animals had the hikers never seen before?",
            "answer": "5"
        },
        {
            "passage": "The firemen responded to an emergency call. They arrived within 5 minutes.",
            "question": "How long did it take for the firemen to arrive?",
            "answer": "5 minutes"
        },
        {
            "passage": "The students were playing basketball in the gym. They had been playing for 2 hours.",
            "question": "How long had the students been playing basketball?",
            "answer": "2 hours"
        },
        {
            "passage": "The family visited the art museum this weekend. They saw at least 6 paintings from famous artists.",
            "question": "How many paintings did the family see from famous artists?",
            "answer": "At least 6"
        }
    ]),

    pd.DataFrame([
        {
            "passage": "The doctor performed surgery at the hospital and then went to the school to pick up her son.",
            "question": "Was the surgery successful?",
            "answer": "Unknown"
        },
        {
            "passage": "As soon as the book was released, it became a New York Times fiction bestseller.",
            "question": "Is the book non-fiction?",
            "answer": "No, Fiction bestseller"
        },
        {
            "passage": "During the presidential election polls last week, Jeff had 15% more votes than John",
            "question": "Were Jack and John running for president?",
            "answer": "Yes, presidential election"
        },
        {
            "passage": "The soccer team won the game 3-2 in the last minute.",
            "question": "What was the final score?",
            "answer": "3-2"
        },
        {
            "passage": "The new restaurant opened in the city center last night and people were queuing up to get in.",
            "question": "Where did the restaurant open?",
            "answer": "City center"
        },
        {
            "passage": "The teacher gave the students a pop quiz and they all failed.",
            "question": "What kind of quiz did the teacher give?",
            "answer": "Pop quiz"
        },
        {
            "passage": "The new laptop was released yesterday with a battery life of up to 10 hours.",
            "question": "How long is the battery life of the laptop?",
            "answer": "Up to 10 hours"
        },
        {
            "passage": "The students are visiting the local museum as part of their history class.",
            "question": "What class are the students taking?",
            "answer": "History"
        },
        {
            "passage": "The movie theater is showing the new superhero film this weekend.",
            "question": "What type of film is being shown at the theater?",
            "answer": "Superhero film"
        },
        {
            "passage": "The store is having a sale on all items this weekend.",
            "question": "When is the sale happening?",
            "answer": "This weekend"
        },
        {
            "passage": "The little girl was walking her dog in the park when she saw a rainbow.",
            "question": "Where was the little girl walking her dog?",
            "answer": "In the park"
        },
        {
            "passage": "The musician was playing the piano in the concert hall.",
            "question": "What instrument was the musician playing?",
            "answer": "Piano"
        },
        {
            "passage": "The family went to the beach for their summer vacation.",
            "question": "Where did the family go for their vacation?",
            "answer": "To the beach"
        },
        {
            "passage": "The fire alarm went off in the building and everyone had to evacuate.",
            "question": "Why did everyone have to leave the building?",
            "answer": "Because the fire alarm went off"
        },
        {
            "passage": "The children were playing in the backyard when a storm started.",
            "question": "Where were the children playing?",
            "answer": "In the backyard"
        },
        {
            "passage": "The store was closed due to the snowstorm.",
            "question": "Why was the store closed?",
            "answer": "Due to the snowstorm"
        },
        {
            "passage": "The students were taking their final exam in the classroom.",
            "question": "Where were the students taking their exam?",
            "answer": "In the classroom"
        }
    ]),

    pd.DataFrame([
        {
            "passage": "According to Biraben, the plague was present somewhere in Italy in every year between 1346 and 1671",
            "question": "Where was the plague present?",
            "answer": "somewhere in Italy"
        },
        {
            "passage": "Jenna's birthday was yesterday evening and at least 10 of her friends attended the party.",
            "question": "How many of Jenna's friends attended?",
            "answer": "at least 10"
        },
        {
            "passage": "Mitsubishi Motor Corp's vehicle sales fell by 42 percent in June",
            "question": "When did Mitsubishi's sales fall?",
            "answer": "June"
        },
        {
            "passage": "The bullies attacked in the elementary school parking lot and then got sent to the teacher's office.",
            "question": "Who or what did the bullies punch?",
            "answer": "Unknown"
        },
        {
            "passage": "The new movie theater located in the mall is only open on weekends.",
            "question": "When is the movie theater open?",
            "answer": "weekends"
        },
        {
            "passage": "The new iPhone has a faster processor and improved battery life.",
            "question": "What is improved in the new iPhone?",
            "answer": "processor and battery life"
        },
        {
            "passage": "The new law requires all businesses to have a smoke detector installed.",
            "question": "What do businesses need to have installed?",
            "answer": "a smoke detector"
        },
        {
            "passage": "The library's hours of operation have changed due to the pandemic.",
            "question": "What has changed about the library?",
            "answer": "hours of operation"
        },
        {
            "passage": "The local park was closed for two weeks due to the storm damage.",
            "question": "How long was the park closed for?",
            "answer": "two weeks"
        },
        {
            "passage": "The board meeting is scheduled for next Saturday.",
            "question": "When is the board meeting?",
            "answer": "next Saturday"
        },
        {
            "passage": "The new restaurant in town offers free delivery for orders over $50.",
            "question": "What does the restaurant offer for orders over $50?",
            "answer": "free delivery"
        },
        {
            "passage": "The class project is due in two weeks.",
            "question": "When is the class project due?",
            "answer": "two weeks"
        },
        {
            "passage": "The city council voted to increase the sales tax by 3%. ",
            "question": "How much did the city council vote to increase the sales tax by?",
            "answer": "3%"
        },
        {
            "passage": "The new park is located on the corner of Main Street and First Avenue.",
            "question": "Where is the new park located?",
            "answer": "on the corner of Main Street and First Avenue"
        },
        {
            "passage": "The mayor's speech will be broadcast live on the local news channel.",
            "question": "Where will the mayor's speech be broadcast?",
            "answer": "on the local news channel"
        },
        {
            "passage": "The new software update is available for download now.",
            "question": "When is the new software update available?",
            "answer": "now"
        },
        {
            "passage": "The school board is offering a free online course for high school students.",
            "question": "Who is the school board offering a free course to?",
            "answer": "high school students"
        },
        {
            "passage": "The store is offering a 10% discount on all purchases this weekend.",
            "question": "What is the store offering this weekend?",
            "answer": "a 10% discount"
        }
    ]),
]


cloze_convertor = InputOutputPrompt(
    input_formatter=lambda x: f"Example: {x['passage']}",
    output_formatter=lambda x: f"Output: {x['question']}",
    input_output_sep="\n",
    example_sep="\n\n",
    required_keys=["question", "passage"],
    instruction=""
)
cloze_examples = [
    pd.DataFrame([
        {
            "passage": "Barrack Obama believes the best novel is Harry Potter.",
            "question": "Barrack Obama believes the best novel is Harry",
        },
        {
            "passage": "The girl invited 12 friends to her birthday party last week.",
            "question": "The girl invited 12 friends to her birthday ",
        },
        {
            "passage": "Apple computers are worse than Dell computers.",
            "question": "Apple computers are worse",
        },
        {
            "passage": "Welcome to New York.",
            "question": "Welcome to New"
        },
        {
            "passage": "Rachel loves to play soccer.",
            "question": "Rachel loves to play"
        },
        {
            "passage": "This car costs $25,000.",
            "question": "This car costs"
        },
        {
            "passage": "Adam likes to watch horror movies.",
            "question": "Adam likes to watch"
        },
        {
            "passage": "I ate a banana for breakfast this morning.",
            "question": "I ate a banana for"
        },
        {
            "passage": "The cat slept all day yesterday.",
            "question": "The cat slept all"
        },
        {
            "passage": "My sister lives in London.",
            "question": "My sister lives in"
        },
        {
            "passage": "The movie theatre is 2 miles away.",
            "question": "The movie theatre is"
        },
        {
            "passage": "We went on a camping trip last week.",
            "question": "We went on a camping"
        },
        {
            "passage": "John won the race yesterday.",
            "question": "John won the"
        },
        {
            "passage": "The restaurant is closed on Monday.",
            "question": "The restaurant is closed"
        },
        {
            "passage": "My brother plays the guitar.",
            "question": "My brother plays"
        },
        {
            "passage": "I took the bus to school today.",
            "question": "I took the bus to"
        },
        {
            "passage": "The store is located on Main Street.",
            "question": "The store is located on"
        },
        {
            "passage": "My dad loves to eat pizza.",
            "question": "My dad loves to eat"
        },
        {
            "passage": "The dog barked all night.",
            "question": "The dog barked all"
        },
        {
            "passage": "The teacher gave us a quiz today.",
            "question": "The teacher gave us a"
        }
    ]),
]

cloze_choices = InputOutputPrompt(
    input_formatter=lambda x: f"Example: {x['example']}\nList alternatives:\n- {x['alternatives1']}\n- {x['alternatives2']}\n- {x['alternatives3']}",
    output_formatter=lambda x: f"",
    input_output_sep="",
    example_sep="\n\n",
    required_keys=["example", "alternatives1", "alternatives2", "alternatives3"],
    instruction="Output a list of unique alternatives for each example.\n\n"
)

cloze_choice_examples = [
    pd.DataFrame([
        {
            "example": "Barrack Obama believes the",
            "alternatives1": "best novel is Harry Potter",
            "alternatives2": "worst book is Harry Potter",
            "alternatives3": "United States is great"
        },
        {
            "example":"The Beatles were honored in:",
            "alternatives1":"Buckingham Palace",
            "alternatives2":"Mexico",
            "alternatives3":"Tower of London"
        },
        {
            "example":"Jerry Baker:",
            "alternatives1":"is part of a soccer team",
            "alternatives2":"is not part of a soccer team",
            "alternatives3":"is a character in a book"
        },
        {
            "example":"The capital of France is:",
            "alternatives1":"Paris",
            "alternatives2":"Amsterdam",
            "alternatives3":"Berlin"
        },
        {
            "example":"The Grand Canyon is located in:",
            "alternatives1":"Colorado",
            "alternatives2":"Arizona",
            "alternatives3":"New Mexico"
        },
        {
            "example":"The longest river in the world is:",
            "alternatives1":"Nile",
            "alternatives2":"Amazon",
            "alternatives3":"Mississippi"
        },
        {
            "example":"The movie Avatar was released in:",
            "alternatives1":"2009",
            "alternatives2":"2005",
            "alternatives3":"2010"
        },
        {
            "example":"The iPhone was released in:",
            "alternatives1":"2007",
            "alternatives2":"2008",
            "alternatives3":"2006"
        },
        {
            "example":"Mount Everest is located in:",
            "alternatives1":"India",
            "alternatives2":"China",
            "alternatives3":"Nepal"
        },
        {
            "example":"The periodic table includes:",
            "alternatives1":"elements",
            "alternatives2":"animals",
            "alternatives3":"plants"
        },
        {
            "example":"The Statue of Liberty is located in:",
            "alternatives1":"New York",
            "alternatives2":"New Jersey",
            "alternatives3":"Connecticut"
        },
        {
            "example":"The capital of the United States is:",
            "alternatives1":"Washington D.C.",
            "alternatives2":"New York",
            "alternatives3":"Los Angeles"
        },
        {
            "example":"The first President of the United States was:",
            "alternatives1":"George Washington",
            "alternatives2":"Thomas Jefferson",
            "alternatives3":"Abraham Lincoln"
        },
        {
            "example":"The Battle of Hastings was fought in:",
            "alternatives1":"1066",
            "alternatives2":"1812",
            "alternatives3":"1314"
        },
        {
            "example":"The currency of the United Kingdom is:",
            "alternatives1":"Euros",
            "alternatives2":"Pounds",
            "alternatives3":"Dollars"
        }
    ])
]

cloze_completion = InputOutputPrompt(
    input_formatter=lambda x: f"Select One Choice:\n1. {x['alternatives1']}\n2. {x['alternatives2']}\n3. {x['alternatives3']}\n\nPassage: {x['passage']}\n\nThe passage \"Passage\" states: {x['statement']}: \"Choice\":",
    output_formatter=lambda x: f"{x['answer']}",
    input_output_sep=" ",
    example_sep="\n\n----\n\n",
    required_keys=["passage", "alternatives1", "alternatives2", "alternatives3", "statement", "answer"],
    instruction="Select one choice from the passage.\n\n"
)

cloze_completion_examples = [
    pd.DataFrame([
        {
            "passage": "Microsoft Corporation produces computer software, consumer electronics, and personal computers. It is headquartered at the Microsoft Redmond campus located in Redmond, Washington, United States.",
            "alternatives1": "consumer electronics",
            "alternatives2": "Play Stations",
            "alternatives3": "cameras",
            "statement": "Microsoft Corporation sells",
            "answer": "consumer electronics"
        },
        {
            "passage":"Sir Elton Hercules John CH CBE is a British singer, pianist and reknowned composer. His nickname is the Rocket Man.",
            "alternatives1":"and tall man",
            "alternatives2":"and trombone player",
            "alternatives3":"and reknowned composer",
            "statement": "Sir Elton John is a musician",
            "answer": "and reknowned composer"
        },
        {
            "passage":"The Mac versus Windows PC debate has been going on for a long time.  Most people say the Windows PC three is superior. It comes down to personal preference.",
            "alternatives1":"Lenovo computers",
            "alternatives2":"Windows PC three",
            "alternatives3":"Dell computers",
            "statement": "Apple computers are superior to",
            "answer": " Windows PC three"
        },
        {
            "passage":"The Amazon rainforest is the largest tropical rainforest in the world. It spans across nine countries in South America.",
            "alternatives1":"Australia",
            "alternatives2":"Africa",
            "alternatives3":"South America",
            "statement": "The Amazon rainforest is located in",
            "answer": "South America"
        },
        {
            "passage":"The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait, the 1-mile-wide, 3-mile-long channel connecting San Francisco Bay and the Pacific Ocean.",
            "alternatives1":"San Francisco Bay and the Atlantic Ocean",
            "alternatives2":"San Francisco Bay and the Pacific Ocean",
            "alternatives3":"San Diego Bay and the Pacific Ocean",
            "statement": "The Golden Gate Bridge connects",
            "answer": "San Francisco Bay and the Pacific Ocean"
        },
        {
            "passage":"The United Nations is an international organization founded in 1945. Its mission is to maintain international peace and security.",
            "alternatives1":"promote human rights",
            "alternatives2":"end poverty",
            "alternatives3":"maintain international peace and security",
            "statement": "The primary mission of the United Nations is to",
            "answer": "maintain international peace and security"
        },
        {
            "passage":"The Apollo 11 mission was the first spaceflight that landed humans on the Moon. It was launched on July 16, 1969, with Neil Armstrong and Edwin Aldrin as the astronauts.",
            "alternatives1":"Neil Armstrong and Michael Collins",
            "alternatives2":"Neil Armstrong and Edwin Aldrin",
            "alternatives3":"Buzz Aldrin and Michael Collins",
            "statement": "The Apollo 11 mission was manned by",
            "answer": "Neil Armstrong and Edwin Aldrin"
        },
        {
            "passage":"The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was built in 1889 as the entrance arch for the 1889 World's Fair.",
            "alternatives1":"Berlin, Germany",
            "alternatives2":"Rome, Italy",
            "alternatives3":"Paris, France",
            "statement": "The Eiffel Tower is located in",
            "answer": "Paris, France"
        },
        {
            "passage":"The Louvre Museum is the world's largest art museum and a historic monument in Paris, France. It is home to many famous works of art such as the Mona Lisa.",
            "alternatives1":"Venice, Italy",
            "alternatives2":"Rome, Italy",
            "alternatives3":"Paris, France",
            "statement": "The Louvre Museum is located in",
            "answer": "Paris, France"
        },
        {
            "passage":"The Taj Mahal is a white marble mausoleum located in Agra, India. It was commissioned by Mughal emperor Shah Jahan in 1632.",
            "alternatives1":"Bangkok, Thailand",
            "alternatives2":"Agra, India",
            "alternatives3":"Beijing, China",
            "statement": "The Taj Mahal is located in",
            "answer": "Agra, India"
        }
    ])
]


class RTEDecomp(Decomposition):
    def __init__(self, task_name, data_dir, val_split="validation"):
        super().__init__(task_name, data_dir, val_split)

    def get_boost_decomp_examples(self, data_train, boost_id):
        if boost_id < 4:
            return [
                questioner_examples[boost_id],
                openended_qa_examples[boost_id],
            ]
        else:
            return [
                cloze_examples[0],
                cloze_choice_examples[0],
                cloze_completion_examples[0]
            ]

    def zero_few_baseline(
        self,
        test_data,
        few_shot_df,
        manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer,
        prompt_suffix="",
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

                icl_str = ""
                if do_few_shot:
                    for s_ind, s_row in few_shot_df.iterrows():
                        icl_str += f"{s_row['inputs_pretokenized']} {s_row['targets_pretokenized']}\n\n\n"

                prompt = f"{icl_str}{{text:}}"
                pmp = prompt.format(text=text)
                if i == 0:
                    print(pmp)

                raw_answer = get_response(
                    pmp,
                    manifest_answer,
                    overwrite=bool(overwrite_manifest_answer),
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
                if is_yes and (not is_no):
                    pred = "True"
                if is_no and (not is_yes):
                    pred = "False"
                elif not is_no and not is_yes:
                    pred = "False"

                    
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
        quesiton_prompt = f"{prompt_suffix}\n\nStatement: {{statement:}}\nQuestion:"
        quesiton_prompt = quesiton_prompt.format(statement=statement).replace("\n\nAnswer:", "\nAnswer:")
        chopped_answer = get_response(
            quesiton_prompt,
            manifest,
            max_toks=50)
        chopped_answer = chopped_answer.split("\n")
        question = [ch for ch in chopped_answer if ch][0]
        answer = [ch for ch in chopped_answer if ch.startswith("Answer: ")]
        if answer:
            answer = answer[0].replace(",", "").replace(".", "").replace("?", "").replace("Answer: ", "")
            answer = " ".join([a for a in answer.split() if a not in stops])
        else:
            answer = ''
        
        if "A:" in question:
            statement = statement.strip(".")
            return f"{statement}. Yes or no?"
        
        # ####################
        # with open(question_file, 'a') as f:
        #     json_string = json.dumps({'prompt': quesiton_prompt, "completion":question})
        #     f.write(json_string + '\n')             
        # ####################           
        return question, answer, quesiton_prompt

    def open_qa(self, question, passage, prompt, boost_ex, manifest, overwrite_manifest):
        prompt_suffix = prompt(boost_ex)
        qa_prompt = f"{prompt_suffix}\n\n----\n\nContext: {{passage:}}\n\nQuestion: {{question:}}\n\nAnswer:"
        qa_prompt = qa_prompt.format(passage=passage, question=question)
        answer = get_response(
            qa_prompt, 
            manifest, 
            max_toks=50
        )
        answer = answer.replace(",", "").replace(".", "").replace("?", "")
        answer = [a for a in answer.split("\n") if a]
        if answer:
            answer = answer[0]
        else:
            answer = passage
        return answer, qa_prompt

    def resolve_pred(self, answer, open_answer):
        answer = answer.lower()
        is_yes = "yes" in answer.split() or "true" in answer.split()
        is_no = "no" in answer.split()  or "false" in answer.split()
        is_maybe = False
        answer = answer.replace("-", "")
        
        pred = "False" 
        if is_yes and (not is_maybe and not is_no) or (answer in open_answer or open_answer in answer):
            pred = "True"
        if is_no and (not is_maybe and not is_yes):
            pred = "False"
        return pred

    def get_choices_answer(self, chopped_answer, cuttoff, prompt, boost_ex, manifest, get_choices_prompt=''):
        prompt_suffix = prompt(boost_ex)
        prompt = f"{prompt_suffix}\n\nExample: {{example:}}\nList alternatives:\n- {{cuttoff:}}\n"
        choices_answer = get_response(
            prompt.format(example=chopped_answer, cuttoff=cuttoff), 
            manifest,
            max_toks = 30
        )
        choices_answer = choices_answer.split("\n\n")[0]
        choices_answer = choices_answer.split("\n")
        choices_answer = [a.replace("- ", "").strip() for a in choices_answer]
        choices_answer = [a for a in choices_answer if cuttoff.lower() not in a.lower()] 
        choices_answer = list(sorted(set(choices_answer)))
        choices_answer = choices_answer[:min(len(choices_answer), 2)]
            
        choices_answer = list(sorted(set(choices_answer)))
        choices_answer.append(cuttoff)
        choices_answer = [ch.strip(".") for ch in choices_answer]
        return choices_answer, prompt
    
    def get_chopping(self, question, prompt, boost_ex, manifest, cuttoff_size=2, chopper_prompt=''):
        prompt_suffix = prompt(boost_ex)
        prompt = f"{prompt_suffix}\n\nExample: {{question:}}\nOutput:"
        chopped_answer = get_response(
            prompt.format(question=question), 
            manifest, 
            max_toks = len(question.split())*4
        )
        chopped_answer = chopped_answer.split("\n")[0]  
        chopped_list = chopped_answer.split()
        question = question.split()
        cuttoff = [t for t in question if t not in chopped_list]
            
        cuttoff_str = " ".join(cuttoff).strip(".")
        chopped_list_str = " ".join(chopped_list).strip(".")
        if not cuttoff or chopped_list_str.endswith(cuttoff_str):
            chopped_list = question[0:-cuttoff_size]
            cuttoff = question[-cuttoff_size:]
        cuttoff = " ".join(cuttoff)
        chopped_answer = " ".join(chopped_list)
        cuttoff = cuttoff.strip(".")
        return chopped_answer, cuttoff, prompt

    def get_final_selection(self, choices_answer, passage, chopped_answer, prompt, boost_ex, manifest, selector_prompt=''):
        prompt_suffix = prompt(boost_ex)
        select_choice_str = ""
        gold_choice = choices_answer[-1]
        other_choices = choices_answer[:-1]
        for num, ch in enumerate(choices_answer):
            select_choice_str += f"\n{num+1}. {ch}"
        prompt = f"{prompt_suffix}\n\n----\n\nSelect one Choice:{{choices_str:}}\n\nPassage: {{passage:}}\nThe passage \"Passage\" states: {{chopped_answer:}} \"Choice\": "
        
        select_answer = get_response(
            prompt.format(choices_str=select_choice_str, passage=passage, chopped_answer=chopped_answer), 
            manifest, 
            max_toks = max(len(c.split()) for c in choices_answer)
        )
        select_answer = select_answer.lower()
        select_answer = select_answer.split("\n")[0].strip(".")
        
        if select_answer.lower() in gold_choice.lower():
            answer = "True"
        else:
            answer = "False"
        return answer, prompt

    def run_decomposed_prompt(
        self, test_data, boost_data_train, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer
    ):
        expt_log, all_boost_preds, labels = self._run_decomp_single_data(test_data, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer)
        expt_log_train, all_boost_train_preds, train_labels = self._run_decomp_single_data(boost_data_train, boost_dfs, manifest_question, manifest_answer, overwrite_manifest_question, overwrite_manifest_answer, run_limit=1000)
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



        for i, (ind, row) in tqdm(
            enumerate(test_data.iterrows()), total=len(test_data)
        ):
            input = row['inputs_pretokenized']
            gold = row['targets_pretokenized']
            passage = input.split("Question: ")[0].strip("\n")
            statement = input.split("Question: ")[-1].replace("True or False?", "")

            if i == run_limit:
                break

            prompts_across_boost = []
            preds_across_boost = []
            for boost_num, boost_examples in enumerate(boost_dfs):
                all_prompts = []

                if boost_num < 4:
                    question, proposed_answer, question_final_prompt = self.get_question(
                        statement, questioner, boost_examples[0], manifest_question, overwrite_manifest_question
                    )
                   
                    if i == 0:
                        print("PROMPT:")
                        print(question_final_prompt)

                    open_answer, answer_final_prompt = self.open_qa(
                        question, passage, openended_qa, boost_examples[1], manifest_answer, overwrite_manifest_answer
                    )
                    if i == 0:
                        print("\nPROMPT:")
                        print(answer_final_prompt)
                    all_prompts.append(question_final_prompt)
                    all_prompts.append(answer_final_prompt)

                    open_answer = open_answer.replace("-", "")
                    open_answer = " ".join([a for a in open_answer.split() if a not in stops])
                    if proposed_answer:
                        answer = proposed_answer.replace("-", "")
                        answer = " ".join([a for a in answer.split() if a not in stops])
                        if all(wd in open_answer.lower() for wd in answer.lower().split()) or all(wd in answer.lower() for wd in open_answer.lower().split()):
                            pred = "True"
                        else:
                            pred = 'False'
                        if not answer.strip():
                            pred = 'False'
                    else:
                        pred = self.resolve_pred(open_answer.lower(), open_answer)
                else:
                    chopped_answer, cuttoff, chopper_prompt = self.get_chopping(
                        statement, cloze_convertor, boost_examples[0], manifest_question, cuttoff_size=2)

                    choices_answer, choices_prompt = self.get_choices_answer(
                        chopped_answer, cuttoff, cloze_choices, boost_examples[1], manifest_question)

                    pred, selector_prompt = self.get_final_selection(
                        choices_answer, passage, chopped_answer, cloze_completion, boost_examples[2], manifest_answer)

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
    args.num_boost = 5
    task_name = "super_glue_rte"
    data_dir = f"{DATA_DIR}/P3/data_feather/super_glue_rte_GPT_3_style/"
    decomp = RTEDecomp(task_name, data_dir)
    decomp.run(args)


if __name__ == "__main__":
    main()
