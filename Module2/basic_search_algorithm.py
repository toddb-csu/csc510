# Todd Bartoszkiewicz
# CSC510: Foundations of Artificial Intelligence
# Module 2: Porfolio Milestone
#
# Basic search algorithm
#
# This Python program will search for a number in a list of numbers.
#
def linear_search(my_array, search_item):
    for i in range(len(my_array)):
        if my_array[i] == search_item:
            return True
    # If the item isn't found, return false so we know the item isn't in the list
    return False


def print_result(found, my_string):
    if found:
        print(f"Found {my_string} in the string array")
    else:
        print(f"{my_string} was NOT found in the string array")


if __name__ == '__main__':
    text_array = ['CSC', '510', 'Foundations', 'Artificial', 'Intelligence']

    # True test
    search_string = 'Artificial'
    result = linear_search(text_array, search_string)
    print_result(result, search_string)

    # False test
    search_string = 'Machine'
    result = linear_search(text_array, search_string)
    print_result(result, search_string)
