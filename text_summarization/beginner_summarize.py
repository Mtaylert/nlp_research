import spacy
import textwrap
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import textwrap




class ExtractiveSummarization:

    def __init__(self, text, stopwords, punctuation, reduction_rate = 0.1):
        self.stopwords = stopwords
        self.punctuation =  punctuation
        self.punctuation += '\n'
        self.text = text
        self.nlp_pipe = spacy.load('en_core_web_sm')
        self.document = self.nlp_pipe(text)
        self.tokens = [token.text for token in self.document]
        self.reduction_rate = reduction_rate
        self.word_frequences = self.get_word_frequencies()
        self.sentence_tokens = [sent for sent in self.document.sents]
        self.sentence_scores = self.get_sentence_scores()
        self.summarization = self.summarize()

    def get_word_frequencies(self):
        word_frequencies = {}
        for word in self.document:
            if word.text.lower() not in self.stopwords:
                if word.text.lower() not in self.punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1

        max_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        return word_frequencies

    def get_sentence_scores(self):
        sentence_scores = {}
        for sent in self.sentence_tokens:
            word_count = 0
            for word in sent:
                if word.text.lower() in self.word_frequences.keys():
                    word_count += 1
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = self.word_frequences[word.text.lower()]
                    else:
                        sentence_scores[sent] += self.word_frequences[word.text.lower()]
        return sentence_scores

    def summarize(self):
        summary_length = int(len(self.sentence_scores) * self.reduction_rate)
        summary = nlargest(summary_length, self.sentence_scores, key=self.sentence_scores.get)
        final_summary = [word.text for word in summary]
        summary = ' '.join(final_summary)
        return summary


text = """Just last week, the new-look Los Angeles Lakers appeared to have things on track. They'd won five of six games after starting the season 0-2, a stretch that was capped off by a win in which the star trio of LeBron James, Anthony Davis and Russell Westbrook combined for 84 points, while Carmelo Anthony chipped in 15 off the bench. But things quickly took a turn for the worse; James missed the next two games, both losses, and the Lakers are now .500 and tied for eighth in the West. 
This team is chock full of superstar talent and has championship aspirations, but it's unclear if they can reach that level. Their minus-1.2 scoring differential ranks 18th in the league and that's come against the second-easiest schedule in the NBA so far this season.
It remains to be seen how well James, Davis, Westbrook and the supporting cast will mesh on the court, and if they have enough to compete at the highest levels when the schedule gets tougher. With that in mind, let's look at some early season data and go through four big questions that will help determine whether this Lakers' season will end in glory or in a high-profile disappointment.
James arrived in Los Angeles in 2018 having reached the Finals in each of the eight previous seasons. That incredible streak was snapped during his first season as a Laker in part because of the Christmas Day groin injury that cost him more than a month of the season. After James and the Lakers won a title in the bubble in 2020, the injury bug returned when Solomon Hill rolled into James' ankle, undermining last season's title defense.
King James was once the iron man of pro basketball, but that is no longer the case. An ankle injury already sidelined him for two games this season and he's missed two more with an abdominal strain that will keep him out until at least Friday. While James has been relatively healthy throughout his career, Davis has a lengthy injury history, including a strained groin that hampered him during the Lakers' first-round playoff loss to the Phoenix Suns in June.
The numbers are clear: if James and Davis are healthy, this squad can be trouble for the rest of the league. If they are not, the Lakers are headed for mediocrity.
Since 2019-20, the Lakers are 85-32 in games when both Davis and James play. That's a winning percentage of 73%; for comparison, the Milwaukee Bucks, the NBA's best regular-season team in that span, have won 69% of their games. However, in games when James has played without Davis, that drops to 59%. In games where Davis has played without James it drops to 38%.
The Lakers have looked like contenders when they have both James and Davis available, and they are much less scary when they don't. The best ability in the NBA is availability, which is especially true for James and Davis this season."""


summary = ExtractiveSummarization(text, stopwords=list(STOP_WORDS), punctuation=punctuation).summarization
print(summary)