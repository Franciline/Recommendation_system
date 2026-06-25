import pandas as pd
import numpy as np
import ollama


def assign_batch_number(df, max_length):
    batch_number = 0
    current_length = 0
    batch_numbers = []

    for length in df['Length']:
        if current_length + length > max_length:
            batch_number += 1
            current_length = 0  # Reset the current length
        batch_numbers.append(batch_number)
        current_length += length

    return batch_numbers


def call_model_by_batch(comments_batched: pd.DataFrame, prompt_type: str) -> str:
    prompt_types = {
        "user": f"""You will receive multiple comments from a single user about different board games.
        These comments are in French and are separated by newlines.
        Your goal is to summarize the user’s overall preferences by identifying what they liked and disliked across all the games.""",

        "game": f"""You will receive multiple comments from different users about a single board game.
        These comments are in French and are separated by newlines.
        Your goal is to summarize the game's overall strengths and weaknesses based on these comments
        """,

        "generate_comment": """You are an expert in analyzing user reviews and generating realistic
        comments based on similar opinions. I will give you comments from different users about the board game (comments by different usersare separated by newlines)
        Based on these comments, generate a new user comment that: feels natural and realistic, reflects the general sentiment and themes from the given comments and
        introduces slight variations in style and phrasing to sound unique. Here are users comments : """,

        """descriptions""": """You will receive multiple board game descriptions in French. Each description is separated by a newline.
        Your task is to summarize these descriptions and to identify key mechanics, target audience, qualit, complexity etc.
        The goal is to find what is in common for all these games. Here are the descriptions: """,

        "desc_summary": """You will receive descriptions in French on 10 board games. Each description is separated by a newline.
        Your task is to summarize all these descriptions one by one. Here are the descriptions: """,

        "comment_summary": """Tu vas recevoir l'ensemble de commentaires sur un seul jeu de société. Ton but 
        c'est de faire un résumé court. Chaque commentaire est séparé par newline. Voici les commentaires : """,

        "combine_phrases": """Tu vas recevoir plusieurs phrases, chaque phrase est séparé par le saut de lignt.
        Ton but c'est de les combiner dans un seul commentaire qu'un utilisateur pourrait donner sur un jeu de société.
        Le commentaire ne doit pas être très long, 
        mais il doit inclure les points importants (bien décrire le jeu). Tu ne dois pas ajouter des explications, donne juste le commentaire généré à partir des phrases.
        Voici des phrases : """,

    }

    prompt = prompt_types[prompt_type]
    messages = [{"role": "user", "content": prompt + comments_batched[0]}]
    messages += [{"role": "user", "content": comments_batched[i]} for i in range(1, len(comments_batched))]
    print(len(messages))
    response = ollama.chat(model='llama3.2', messages=messages)
    print("Nb tokens:", response["prompt_eval_count"])
    return response


def calc_nb_words(comments: pd.DataFrame) -> int:
    return comments.str.split().explode().size
