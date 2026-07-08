def levenshtein(a, b):

    m = len(a)

    n = len(b)

    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0]=i

    for j in range(n+1):
        dp[0][j]=j

    for i in range(1,m+1):

        for j in range(1,n+1):

            cost = 0 if a[i-1]==b[j-1] else 1

            dp[i][j]=min(

                dp[i-1][j]+1,

                dp[i][j-1]+1,

                dp[i-1][j-1]+cost

            )

    return dp[m][n]


print("harga -> harag")

print(levenshtein("harga","harag"))

print("packing -> paking")

print(levenshtein("packing","paking"))