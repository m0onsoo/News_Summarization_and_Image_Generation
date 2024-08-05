# News-cralwer-summarization-image_generator bot

## How it works

1. Collect the title and body of news articles from [CBS NEWS](https://www.cbsnews.com/us/).
2. Briefly summarizes the content of the article so that people can easily understand it.
3. Based on the summary, generative AI creates news thumbnails.

## Getting Started

```
# Installation

git clone https://github.com/m0onsoo/news-summary-image-generator.git
cd news-summary-image-generator/source
```

First, you need to gather the news data from the website.
```
python 0_crawling/cbs_crawling.py
```
Additionally, summarization and image are generated.
```
python 1_sentence/summarize.py
```

## Future work

+ Automatic classification of important and popular topics among news articles
+ Instagram automatic upload bot to be developed
