import hashlib

target_hash = "9edf6713ae21b1a80951c3e9aa069a0213a7a4eb"

# Common wordlist (you can replace with rockyou.txt)
wordlist = ["password", "admin", "hacker", "hello", "india"]

for word in wordlist:
    hashed = hashlib.sha1(word.encode()).hexdigest()
    
    if hashed == target_hash:
        print("Found:", word)
        break