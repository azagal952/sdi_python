## Function 1 : course_comment

def course_comment(course):
    course_lower = course.lower()

    if course_lower in ["maths", "python"]: return "This is very useful!"
    elif course_lower == "meditation": return "How nice!"
    elif course_lower == "magic": return "You're not at Hogwarts"
    else: return "What is this COURSE?"

def test_course_comment_maths():
    assert course_comment("MATHs") == "This is very useful!"

def test_course_comment_logistics():
    assert course_comment("LOGISTICS") == "What is this COURSE?"

## Function 2 : is_unique

def is_unique(liste):
    """
    Checks if there are duplicate items in a list.
    Args:
        liste: the input list.

    Returns: True if there are duplicate items, False otherwise.

    """
    return len(liste) == len(set(liste))

def test_is_unique_false():
    assert is_unique([1,1,2,3]) == False

def test_is_unique_true():
    assert is_unique([1,2,3,6]) == True

## Function 3 : triangle_shape

def triangle_shape(height):
    """
    Provides a string in a shape of a triangle.
    Args:
        height: the height of the triangle.

    Returns: the triangle String.

    """
    if height == 0: return ""

    triangle = []
    for i in range(height):
        num = 2 * i + 1
        num_spaces = height - i - 1
        triangle.append(" " * num_spaces + "x" * num)

    return "\n".join(triangle)

def test_triangle_shape_6():
    assert triangle_shape(6) =="     x\n    xxx\n   xxxxx\n  xxxxxxx\n xxxxxxxxx\nxxxxxxxxxxx"