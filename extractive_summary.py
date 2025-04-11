import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

def summarize():
    text = """
    Importance of Trees
    Trees are important to us in a lot of ways and we cannot ignore their importance. They are important because they give us fresh air to breathe, food to eat and shelter/shade from sunlight and rainfall. Besides this, there are many medicines in the market that are made up of trees extracts. Apart from this, there are plants and trees that have medicinal value.

    They bring peacefulness; create a pleasing and relaxing environment. Also, they help in reflecting the harmful rays of the sun and maintaining a balanced temperature. Besides, they also help in water conservation and preventing soil erosion. They also manage the ecosystem and from ancient times several varieties of plants are worshipped.



    Benefits of Trees
    Trees provide us many benefits some of which we can’t see but they make a huge difference. They help in fighting back the climate changes by absorbing greenhouse gases which are the main cause of climate change.

    Moreover, they replenish groundwater and filter the air from harmful pollutants and odors. Besides, they are a great source of food and the king of fruits ‘Mango’ also grow on trees.



    Moreover, they are the cause of rainfall as they attract clouds towards the surface and make them rain. They can be teachers, playmates and a great example of unity in diversity.

    Above all, they are a good source of reducing air, water, and noise pollution.

    Value of Trees
    When a seed of a plant or tree grow it makes the area around it greener. Also, it supports many life forms. Birds make their nests, many reptiles and animals live on it or near it.

    Besides, all these many beautiful flowers, food growing on it. Moreover, many parts of trees such as roots, leaves, stem, flower, seeds, are also edible. Most importantly they never ask anything in return for their services and the gifts they give. Trees also keep the balance in the ecosystem and ecology.

    To conclude, we can say that trees are very important and beneficial for every life form on earth. Without them, the survival of life on earth will become difficult and after some time every species starts to die because of lack of oxygen on the planet. So, to save our lives and to survive we have to learn the importance of trees and also have to teach our children the importance of trees."""

    stopwords = list(STOP_WORDS)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    tokens = [token.text for token in doc]

    freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in freq.keys():
                freq[word.text] = 1
            else:
                freq[word.text] += 1

    max_freq = max(freq.values())

    for word in freq.keys():
        freq[word] = freq[word]/max_freq

    sent_tokens = [sent for sent in doc.sents]

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = freq[word.text]
                else:
                    sent_scores[sent] += freq[word.text]

    select_len = int(len(sent_tokens) * 0.3)

    summary = nlargest(select_len, sent_scores, key = sent_scores.get)

    final = [word.text for word in summary]
    summary = ' '.join(final)

    print(summary, doc, len(text.split(' ')), len(summary.split(' ')))
    return summary, doc, len(text.split(' ')), len(summary.split(' '))