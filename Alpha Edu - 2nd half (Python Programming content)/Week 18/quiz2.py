def kronecker_product(matrix1, matrix2):
    result = []
    for row1 in matrix1:
        for row2 in matrix2:
            new_row = []
            for elem1 in row1:
                for elem2 in row2:
                    new_row.append(elem1 * elem2)
            result.append(new_row)
    return result

# Example usage
A = [[1, 2, 3]]
B = [[0, 1, 0]]
result = kronecker_product(A, B)
for row in result:
    print(row)
