"""Text cleaning for noisy, code-mixed social-media text (English, Hindi, Hinglish).

The same function is used in training (notebook) and in the app, so the model
always sees text cleaned in the same way.
"""
import re
import unicodedata

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")
RT_PREFIX = re.compile(r"^\s*RT\s+", re.IGNORECASE)
LEADING_JUNK = re.compile(r"^[\s:,\-]+")
REPEATED_LETTERS = re.compile(r"([^\W\d_])\1{2,}")      # "sooooo" -> "soo" (letters only, digits stay)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
SPACES = re.compile(r"\s+")


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))   # unify look-alike characters
    text = RT_PREFIX.sub("", text)                    # remove the "RT" retweet marker
    text = URL.sub(" ", text)                         # remove links
    text = MENTION.sub(" ", text)                     # remove @mentions
    text = HASHTAG.sub(r"\1", text)                   # "#FakeNews" -> "FakeNews"
    text = EMOJI.sub(" ", text)                       # remove emojis
    text = REPEATED_LETTERS.sub(r"\1\1", text)        # shorten stretched words
    text = LEADING_JUNK.sub("", text)                 # leftover ": " after removing a mention
    return SPACES.sub(" ", text).strip()              # tidy up spaces
