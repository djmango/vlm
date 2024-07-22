import torch
from train import normalize_coords, unnormalize_coords

def test_normalize_coords_single_image():
    # Set up test data
    x = torch.tensor([
        [100, 200, 300, 400],  # bbox 1
        [500, 600, 700, 800],  # bbox 2
        [1000, 500, 1500, 900]  # bbox 3
    ])  # shape: [3, 4]

    img_sizes = torch.tensor([1920, 1080])  # shape: [2]

    # Call the function
    normalized = normalize_coords(x, img_sizes)

    # Expected output (calculated manually)
    expected = torch.tensor([
        [100/1920, 200/1080, 300/1920, 400/1080],
        [500/1920, 600/1080, 700/1920, 800/1080],
        [1000/1920, 500/1080, 1500/1920, 900/1080]
    ])

    # Assert that the output matches the expected result
    assert torch.allclose(normalized, expected, atol=1e-6), \
        f"Expected {expected}, but got {normalized}"

    print("Test passed successfully!")

def test_normalize_coords_batch():
    # Set up test data for two images
    x = torch.tensor([
        # First image bboxes (same as single image test)
        [
            [100, 200, 300, 400],
            [500, 600, 700, 800],
            [1000, 500, 1500, 900]
        ],
        # Second image bboxes
        [
            [50, 100, 200, 300],
            [300, 400, 600, 700],
            [800, 300, 1100, 650]
        ]
    ])  # shape: [2, 3, 4]

    img_sizes = torch.tensor([
        [1920, 1080],  # First image size
        [1280, 780]    # Second image size
    ])  # shape: [2, 2]

    # Call the function
    normalized = normalize_coords(x, img_sizes)

    # Expected output (calculated manually)
    expected = torch.tensor([
        [
            [100/1920, 200/1080, 300/1920, 400/1080],
            [500/1920, 600/1080, 700/1920, 800/1080],
            [1000/1920, 500/1080, 1500/1920, 900/1080]
        ],
        [
            [50/1280, 100/780, 200/1280, 300/780],
            [300/1280, 400/780, 600/1280, 700/780],
            [800/1280, 300/780, 1100/1280, 650/780]
        ]
    ])

    # Assert that the output matches the expected result
    assert torch.allclose(normalized, expected, atol=1e-6), \
        f"Expected {expected}, but got {normalized}"

    print("Batch test passed successfully!")

def test_unnormalize_coords_single_image():
    # Set up test data
    x = torch.tensor([
        [0.0521, 0.1852, 0.1563, 0.3704],  # bbox 1
        [0.2604, 0.5556, 0.3646, 0.7407],  # bbox 2
        [0.5208, 0.4630, 0.7813, 0.8333]   # bbox 3
    ])  # shape: [3, 4]

    img_sizes = torch.tensor([1920, 1080])  # shape: [2]

    # Call the function
    unnormalized = unnormalize_coords(x, img_sizes)

    # Expected output (calculated using math expressions)
    expected = torch.tensor([
        [0.0521 * 1920, 0.1852 * 1080, 0.1563 * 1920, 0.3704 * 1080],
        [0.2604 * 1920, 0.5556 * 1080, 0.3646 * 1920, 0.7407 * 1080],
        [0.5208 * 1920, 0.4630 * 1080, 0.7813 * 1920, 0.8333 * 1080]
    ])

    # Assert that the output matches the expected result
    assert torch.allclose(unnormalized, expected, atol=1), \
        f"Expected {expected}, but got {unnormalized}"

    print("Unnormalize test passed successfully!")

def test_unnormalize_coords_batch():
    # Set up test data for two images
    x = torch.tensor([
        # First image normalized bboxes
        [
            [0.0521, 0.1852, 0.1563, 0.3704],
            [0.2604, 0.5556, 0.3646, 0.7407],
            [0.5208, 0.4630, 0.7813, 0.8333]
        ],
        # Second image normalized bboxes
        [
            [0.0391, 0.1282, 0.1563, 0.3846],
            [0.2344, 0.5128, 0.4688, 0.8974],
            [0.6250, 0.3846, 0.8594, 0.8333]
        ]
    ])  # shape: [2, 3, 4]

    img_sizes = torch.tensor([
        [1920, 1080],  # First image size
        [1280, 780]    # Second image size
    ])  # shape: [2, 2]

    # Call the function
    unnormalized = unnormalize_coords(x, img_sizes)

    # Expected output (calculated using math expressions)
    expected = torch.tensor([
        [
            [0.0521 * 1920, 0.1852 * 1080, 0.1563 * 1920, 0.3704 * 1080],
            [0.2604 * 1920, 0.5556 * 1080, 0.3646 * 1920, 0.7407 * 1080],
            [0.5208 * 1920, 0.4630 * 1080, 0.7813 * 1920, 0.8333 * 1080]
        ],
        [
            [0.0391 * 1280, 0.1282 * 780, 0.1563 * 1280, 0.3846 * 780],
            [0.2344 * 1280, 0.5128 * 780, 0.4688 * 1280, 0.8974 * 780],
            [0.6250 * 1280, 0.3846 * 780, 0.8594 * 1280, 0.8333 * 780]
        ]
    ])

    # Assert that the output matches the expected result
    assert torch.allclose(unnormalized, expected, atol=1), \
        f"Expected {expected}, but got {unnormalized}"

    print("Batch unnormalize test passed successfully!")

# Run the tests
test_normalize_coords_single_image()
test_normalize_coords_batch()
test_unnormalize_coords_single_image()
test_unnormalize_coords_batch()