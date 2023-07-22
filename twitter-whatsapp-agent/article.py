from newspaper import Article
from typing import TypedDict


class ArticleData(TypedDict):
    title: str
    description: str
    authors: list[str]
    content: str
    url: str
    publish_date: str
    keywords: list[str]
    summary: str


def get_article_data(url: str) -> ArticleData:
    article = Article(url)

    # download text
    article.download()

    # parse main content + extract metadata
    article.parse()

    title = article.title
    meta_data = article.meta_data
    description = meta_data["description"]
    authors = article.authors
    content = article.text
    publish_date = str(article.publish_date)

    # nlp to get keywords + summary
    article.nlp()
    keywords = article.keywords
    summary = article.summary

    return {
        "title": title,
        "description": description,
        "authors": authors,
        "content": content,
        "url": url,
        "publish_date": publish_date,
        "keywords": keywords,
        "summary": summary,
        # "meta_data": dict(meta_data),
    }
