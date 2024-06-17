"""
商と余りを全部足す
1 以上 5 以下の整数 N について、77777 ÷ N の商と余りを求めます。

77777÷1=77777あまり0
77777÷2=38888あまり1
77777÷3=25925あまり2
77777÷4=19444あまり1
77777÷5=15555あまり2
この 5 個の式の商と余りを全て足すと 177595 になります。

77777 + 0 + 38888 + 1 + 25925 + 2 + 19444 + 1 + 15555 + 2 = 177595

それでは、N を 1 以上 600 以下の整数としたときに、77777 ÷ N の商と余りを全て足した値を求めてください。"""

# Set the target number
target = 77777

# Define the range of N
n_min = 1
n_max = 600

# Initialize sum of quotients and remainders
total_sum = 0

# Calculate quotients and remainders for each N, then add them
for N in range(n_min, n_max + 1):
    quotient = target // N
    remainder = target % N
    total_sum += quotient + remainder

print(total_sum)


"""
ひたすら半分増やす
ある整数に対して、「自身の半分の値(小数部切り捨て)を加算する」という操作(結果は必ず整数になる)を「半分増やす」と呼ぶことにする。

42÷2=21なので、42を「半分増やす」と42+21で 63 になる。
63÷2=31.5なので、63を「半分増やす」と63+31で 94 になる。
以上のように、42 を 2 回「半分増やす」と 94 になる。それでは、42 を 32 回「半分増やす」と最後に得られる値は何になりますか。"""
# Set the initial integer
number = 42

# Set the number of operations
operations = 32

# Perform the operation "add half" 32 times
for _ in range(operations):
    number += number // 2

number

"""8 桁以内の全ての 2018 の正の倍数について、その千の位の数字を足し合わせた総和（そうわ）を求めてください。"""
# Find all multiples of 2018 that have up to 8 digits
# Multiply 2018 by every integer from 1 until the largest number that keeps the product under 100000000

# Initialize the sum
sum_thousands_digit = 0

# Loop through the multiples
multiplier = 1
while True:
    product = 2018 * multiplier
    if product >= 100000000:
        break
    # Extract the thousand's place digit
    thousands_digit = (product // 1000) % 10
    sum_thousands_digit += thousands_digit
    multiplier += 1

sum_thousands_digit

"""荷物がいっぱいです
ここに777個の荷物があり、それぞれの重さは1kg、2kg、3kg、……、777kgとなっています。これらを最大積載量5000kgのトラックを何台か使って運ぼうとしています。

トラックに荷物を載せるのに、次のような方針を立てました。

重い荷物から順にトラックに載せていく。
ある荷物を載せると最大積載量を超えてしまう場合は、新しいトラックを用意してそちらに載せる。古いほうのトラックには以降は新しい荷物は載せない。
この方針に従うと、

1台目のトラックには777kg，776kg，775kg，774kg，773kg，772kgの荷物が載せられる。
2台目のトラックには771kg，770kg，769kg，768kg，767kg，766kgの荷物が載せられる。
となります。最終的に何台のトラックが必要になるかを求めてください。"""

# Define the maximum load of a truck
max_load = 5000

# Create a list of packages from 777kg to 1kg
packages = list(range(777, 0, -1))

# Initialize variables to count trucks and track the current load on the current truck
num_trucks = 0
current_load = 0

# Iterate through each package
for package in packages:
    # Check if adding this package would exceed the max load
    if current_load + package > max_load:
        # If yes, this truck is now full, start a new truck
        num_trucks += 1
        current_load = package
    else:
        # Otherwise, add this package to the current truck
        current_load += package

# After the last package, we need to account for the truck currently being used
if current_load > 0:
    num_trucks += 1

num_trucks

"""切手・切手・切手
205円切手が30枚、82円切手が40枚、30円切手が20枚あります。

これらの切手の全部または一部（1枚以上）を使って額面の和として表せる金額は何通りあるか、求めてください。"""

# Define the quantity and value of each stamp type
stamps = {
    205: 30,  # 205 yen stamps, 30 pieces
    82: 40,   # 82 yen stamps, 40 pieces
    30: 20    # 30 yen stamps, 20 pieces
}

# Create a set to hold the possible sums
possible_sums = set()

# Use a nested loop to try all combinations of stamp usage
for num_205 in range(stamps[205] + 1):  # +1 because we include using 0 stamps
    for num_82 in range(stamps[82] + 1):
        for num_30 in range(stamps[30] + 1):
            # Calculate the total value of the selected combination of stamps
            total_value = num_205 * 205 + num_82 * 82 + num_30 * 30
            # Add the total value to the set (sets automatically handle duplicates)
            if total_value > 0:  # Ensure at least one stamp is used
                possible_sums.add(total_value)

# The number of distinct sums that can be made
len(possible_sums)
