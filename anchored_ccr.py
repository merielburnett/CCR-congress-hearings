# stream_ccr.py. need streaming for bad pcs

from datasets import load_dataset
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

# def pos/neg items for each dimension
dimensions = {
    "liberty": (
        [
            "People who are successful in business have a right to enjoy their wealth as they see fit",
            "Society works best when it lets individuals take responsibility for their own lives without telling them what to do.",
            "The government interferes far too much in our everyday lives.",
            "The government should do more to advance the common good, even if that means limiting the freedom and choices of individuals.",
            "Property owners should be allowed to develop their land or build their homes in any way they choose, as long as they don't endanger their neighbors.",
            "I think everyone should be free to do as they choose, so long as they don't infringe upon the equal freedom of others.",
            "People should be free to decide what group norms or traditions they themselves want to follow."
        ],
        [
            "Wealthy individuals should not have unrestricted rights to use their wealth; their financial choices must serve the public interest.",
            "Society works best when authorities set clear rules for individuals rather than leaving them entirely on their own.",
            "The government fails to intervene sufficiently in our everyday lives to safeguard public welfare.",
            "The government should refrain from limiting individual freedom in the name of the common good.",
            "Property owners must adhere to strict development regulations, even if it prevents them from using their land as they wish.",
            "Individuals need well-defined legal constraints on their actions, even if those constraints limit their personal freedom.",
            "People should follow the group norms and traditions determined by community leaders rather than choosing them themselves."
        ]
    ),
    "harm": (
        [
            "Caring for people who have suffered is an important virtue.",
            "I believe that compassion for those who are suffering is one of the most crucial virtues.",
            "We should all care for people who are in emotional pain.",
            "I am empathetic toward those people who have suffered in their lives.",
            "Everyone should try to comfort people who are going through something hard.",
            "It pains me when I see someone ignoring the needs of another human being"
        ],
        [
            "Showing concern for people who have suffered is not an important virtue.",
            "Compassion for those who are suffering is not crucial and often unnecessary.",
            "We need not care for people who are experiencing emotional pain.",
            "I find it unnecessary to empathize with individuals who have suffered.",
            "People should avoid comforting those who are going through hard times.",
            "It does not bother me when someone ignores another person’s needs."
        ]
    ),
    "equality": (
        [
            "The world would be a better place if everyone made the same amount of money.",
            "Our society would have fewer problems if people had the same income.",
            "I believe that everyone should be given the same quantity of resources in life.",
            "I believe it would be ideal if everyone in society wound up with roughly the same amount of money.",
            "When people work together toward a common goal, they should share the rewards equally, even if some worked harder on it.",
            "I get upset when some people have a lot more money than others in my country."
        ],
        [
            "The world functions best when people have different levels of income based on their efforts.",
            "Society would not face fewer problems if incomes were made equal.",
            "Everyone does not need to receive the same resources in life.",
            "It is not ideal for everyone to end up with roughly the same amount of money.",
            "Even if some work harder, rewards should not necessarily be split equally among collaborators.",
            "I do not feel upset when people have large disparities in wealth in my country."
        ]
    ),
    "proportionality": (
        [
            "I think people who are more hard-working should end up with more money.",
            "I think people should be rewarded in proportion to what they contribute.",
            "The effort a worker puts into a job ought to be reflected in the size of a raise they receive.",
            "It makes me happy when people are recognized on their merits.",
            "In a fair society, I want people who work harder than others to end up richer than others.",
            "I feel good when I see cheaters get caught and punished."
        ],
        [
            "Hard work should not guarantee higher earnings.",
            "People should not necessarily be rewarded based on their contributions.",
            "The size of a raise need not reflect the effort a worker puts into a job.",
            "I do not find it satisfying when people are recognized solely on their merits.",
            "Even in a fair society, those who work harder do not need to end up richer than others.",
            "I feel indifferent when I see cheaters get caught and punished."
        ]
    ),
    "loyalty": (
        [
            "I think children should be taught to be loyal to their country.",
            "It upsets me when people have no loyalty to their country.",
            "Everyone should love their own community.",
            "Everyone should defend their country, if called upon.",
            "Everyone should feel proud when a person in their community wins in an international competition.",
            "I believe the strength of a sports team comes from the loyalty of its members to each other."
        ],
        [
            "Children should not be taught to be loyal to their country.",
            "I do not feel upset when people show no loyalty to their country.",
            "It is not necessary for everyone to love their own community.",
            "People should not be obligated to defend their country if called upon.",
            "It is not important to feel proud when someone from your community wins internationally.",
            "A sports team’s strength does not depend on its members’ loyalty to each other."
        ]
    ),
    "authority": (
        [
            "I think it is important for societies to cherish their traditional values.",
            "I feel that most traditions serve a valuable function in keeping society orderly.",
            "I think obedience to parents is an important virtue.",
            "We all need to learn from our elders.",
            "I believe that one of the most important values to teach children is to have respect for authority.",
            "I think having a strong leader is good for society."
        ],
        [
            "It is not important for societies to cherish their traditional values.",
            "Most traditions do not serve a valuable function in keeping society orderly.",
            "Obedience to parents is not an important virtue.",
            "We do not need to learn from our elders.",
            "Teaching children to respect authority is not one of the most important values.",
            "Having a strong leader is not necessarily good for society."
        ]
    ),
    "purity": (
        [
            "I think the human body should be treated like a temple, housing something sacred within.",
            "I believe chastity is an important virtue.",
            "It upsets me when people use foul language like it is nothing.",
            "If I found out that an acquaintance had an unusual but harmless sexual fetish I would feel uneasy about them.",
            "People should try to use natural medicines rather than chemically identical human-made ones.",
            "I admire people who keep their virginity until marriage."
        ],
        [
            "I do not think the human body should be treated like a temple or anything sacred.",
            "I do not believe chastity is an important virtue.",
            "It does not bother me when people use foul language casually.",
            "I would not feel uneasy if I learned someone had a harmless but unusual sexual fetish.",
            "People need not prefer natural medicines over chemically identical human-made ones.",
            "I do not admire people for keeping their virginity until marriage."
        ]
    ),
}

# load SBERT model and build anchors
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
anchors = {}
for dim, (pos_items, neg_items) in dimensions.items():
    pos_emb = model.encode(pos_items, convert_to_numpy=True)
    neg_emb = model.encode(neg_items, convert_to_numpy=True)
    anchors[dim] = pos_emb.mean(axis=0) - neg_emb.mean(axis=0)

# prep monthly aggregator
agg = defaultdict(lambda: {d: [0.0, 0] for d in anchors})

# prep. Stream & score
ds = load_dataset(
    "Eugleo/us-congressional-speeches-subset",
    split="train",
    streaming=True
)

for ex in tqdm(ds, total=5038919, desc="Streaming & scoring"):
    raw_date = ex["date"]
    # parse date
    if isinstance(raw_date, str):
        try:
            dt = datetime.fromisoformat(raw_date)
        except ValueError:
            dt = datetime.fromisoformat(raw_date.rstrip("Z"))
    else:
        dt = raw_date
    # filter years
    if dt.year < 1910 or dt.year > 2020:
        continue
    # drop only truly "Unknown"
    if ex.get("speaker") == "Unknown":
        continue

    month = dt.strftime("%Y-%m")
    text = ex.get("text", "")
    if not text:
        continue

    # embed once
    emb = model.encode([text], convert_to_numpy=True)[0]

    # compute and accumulate each sim
    for dim, anchor in anchors.items():
        sim = cosine_similarity(emb.reshape(1, -1),
                                anchor.reshape(1, -1))[0, 0]
        agg[month][dim][0] += sim
        agg[month][dim][1] += 1

# 5. Build summar y DataFrame
rows = []
for month, metrics in agg.items():
    row = {"month": month}
    for dim, (tot, cnt) in metrics.items():
        row[f"{dim}_avg"] = (tot / cnt) if cnt > 0 else None
    rows.append(row)

df = pd.DataFrame(rows).sort_values("month")
df.to_csv("monthly_ccr_scores_1910_2020.csv", index=False)
print("✓ Saved monthly CCR scores → monthly_ccr_scores_1910_2020.csv")
