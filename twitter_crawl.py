import tweepy, csv, time
import pandas as pd

#"""
consumer_key = 'GSDyidvmJDvBMlGsbwXd5oJcr'
consumer_secret = '1JzMV9nFhtX2WyRtpqNZbsDQ8iIiApkfveiKrASi5uXuIy5wb3'
access_token = '342602156-blKsnAaObTRsuVifwvwSrO3oeaUv3qS1RtoR49Vb'
access_token_secret = '20CUaxbGK91YMOTimOnV3TXnky4ahaKooI4XAoZQoqTkA'
#"""
'''
consumer_key = 'vU17b7Kb18pZlgjx9Oc43aWEj'
consumer_secret = 'Oh5CiY5bp1nULnYD3kl6Z5i6uxk1i8oLpANynncT4L8goni4cn'
access_token = '342602156-ojXdIFyC1VH4aBCaNhcUbSq8QE6Epg86IXCQm8MV'
access_token_secret = 'WNCBXDrP1dx2HCCGXHVlzcTNIwBzRbEvbng8746Cv6cbK'
'''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

tweets = []
query = "hate"

print("starting crawl:",query)

try:
    for tweet in tweepy.Cursor(api.search,q="#"+query,lang="en",since="2017-01-01").items(2000):
        text = tweet.text.replace("&amp;","&").replace(",","").replace("RT","")
        print(text)
        tweets.append(text)
        time.sleep(1e-3)
    pd.DataFrame(tweets).to_csv(query+".csv")
except Exception as e:
    print(e)
    pd.DataFrame(tweets).to_csv(query+".csv")
