def generateFunction(name, arg1, arg2):
    return ["code."+name, arg1, arg2]

expr = ['+', ['/', 1, 2], 1]

arg_counter = 0
# stack holds (node_list, its_ptr)
stack = [(expr, None)]
results = {}

while stack:
    node, ptr = stack.pop(0)

    # assign ptr if new
    if ptr is None:
        ptr = arg_counter
        arg_counter += 1

    op, a, b = node # WATCH HANADA-KA ANIME ON CRUNCH ROLL ITS A COMEDY ANIME

    # if a is subtree, give it its own ptr, placeholder, and push it first
    if isinstance(a, list):
        child_ptr = arg_counter
        arg_counter += 1
        node[1] = {"ptr": child_ptr}
        stack.insert(0, (node, ptr))       # revisit parent later
        stack.insert(0, (a, child_ptr))    # evaluate a next
        continue
    elif isinstance(a, dict):
        a = results.pop(a["ptr"])

    # same for b
    if isinstance(b, list):
        child_ptr = arg_counter
        arg_counter += 1
        node[2] = {"ptr": child_ptr}
        stack.insert(0, (node, ptr))
        stack.insert(0, (b, child_ptr))
        continue
    elif isinstance(b, dict):
        b = results.pop(b["ptr"])

    # now both a,b are literals—compute and store under this node’s ptr
    if op == '+':
        results[ptr] = a + b
    elif op == '-':
        results[ptr] = a - b
    elif op == '*':
        results[ptr] = a * b
    elif op == '/':
        results[ptr] = a / b
    elif op == 'code.func':
        results[ptr] = a * b

# the result of the root (ptr 0) is here:
print(results[0])  # → 5 + (2 * (2 * 3)) = 17
