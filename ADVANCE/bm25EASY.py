from rank_bm25 import BM25Okapi

#Create Documents:
Documents=[
    "Apple is red in colour.",
    "Banana is yellow in colour.",
    "Strawberry is red in colour.",
    "Mango and banana both are yellow in colour."
]

#Tokenise Documents
#  Documents are needed to tokenise so that computer can individually process them
tokens=[]
for doc in Documents:
    t=doc.lower().split()
    tokens.append(t)
bm25= BM25Okapi(tokens)

print("TOKENISED DOCUMENTS:")
for t in tokens :
    print(t)

#Write queries and tokenise them also 
query1="apple mango both colour "
token_query1=query1.lower().split()
print("TOKENISED QUERY 1:")
print(token_query1)

query2="Strawberry is red in colour."
token_query2=query2.lower().split()
print("TOKENISED QUERY 2:")
print(token_query2)

query3="Carrot"
token_query3=query3.lower().split()
print("TOKENISED QUERY 3:")
print(token_query3)

## print bm25 scores
#### It will give an array of all documents with some numeric values that 
#indicated which document is highly relevant

score1=bm25.get_scores(token_query1)
score2=bm25.get_scores(token_query2)
score3=bm25.get_scores(token_query3)
print("SCORE FOR QUERY 1: ",score1)
print("SCORE FOR QUERY 2: ",score2)
print("SCORE FOR QUERY 3: ",score3)
