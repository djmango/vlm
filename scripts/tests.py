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

def batched_cdist_minkowski(XA: torch.Tensor, XB: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """
    Compute the pairwise Minkowski distance between two batched sets of points.

    Args:
    XA (torch.Tensor): First set of points with shape (b, n, m) where b is the batch size,
                       n is the number of points, and m is the dimension of each point.
    XB (torch.Tensor): Second set of points with shape (b, k, m) where b is the batch size,
                       k is the number of points, and m is the dimension of each point.
    p (float): The order of the norm. Default is 2.0 (Euclidean distance).

    Returns:
    torch.Tensor: A tensor of shape (b, n, k) containing the pairwise distances for each batch.

    Note:
    This function is a batched equivalent to scipy.spatial.distance.cdist(XA, XB, metric='minkowski', p=p)
    """
    if p <= 0:
        raise ValueError("p must be greater than 0")

    # Ensure inputs are at least 3D (batch, points, dimensions)
    if XA.dim() == 2:
        XA = XA.unsqueeze(0)
    if XB.dim() == 2:
        XB = XB.unsqueeze(0)

    if XA.dim() != 3 or XB.dim() != 3:
        raise ValueError("Input tensors must be 3D (batch, points, dimensions)")

    # Check if batch sizes match
    if XA.size(0) != XB.size(0):
        raise ValueError("Batch sizes of XA and XB must match")

    # Compute pairwise differences
    diff = XA.unsqueeze(2) - XB.unsqueeze(1)

    # Compute the p-th power of the absolute differences
    distances = torch.abs(diff).pow(p)

    # Sum along the last dimension and take the p-th root
    distances = distances.sum(dim=-1).pow(1/p)

    return distances


def test_cdist_equivalence():
    torch.manual_seed(42)  # For reproducibility

    test_cases = [
        ((2, 3, 4), (2, 5, 4)),  # Batched case
        ((1, 10, 3), (1, 15, 3)),  # Single batch
        ((5, 7, 2), (5, 6, 2)),  # Multiple batches
        ((3, 100, 5), (3, 80, 5)),  # Larger number of points
    ]

    for xa_shape, xb_shape in test_cases:
        XA = torch.rand(*xa_shape)
        XB = torch.rand(*xb_shape)

        # Compute distances using our function
        our_distances = batched_cdist_minkowski(XA, XB, p=1.0)

        # Compute distances using torch.cdist
        torch_distances = torch.cdist(XA, XB, p=1.0)

        # Check if the results are close
        is_close = torch.allclose(our_distances, torch_distances, rtol=1e-5, atol=1e-8)
        max_diff = torch.max(torch.abs(our_distances - torch_distances))

        print(f"Test case: XA shape {xa_shape}, XB shape {xb_shape}")
        print(f"Results match: {is_close}")
        print(f"Max difference: {max_diff}")
        print("--------------------")

        # Assert that the results are close
        assert is_close, f"Results don't match for shapes {xa_shape} and {xb_shape}"

    print("All tests passed successfully!")

# Run the tests
test_cdist_equivalence()
# Run the tests
test_normalize_coords_single_image()
test_normalize_coords_batch()
test_unnormalize_coords_single_image()
test_unnormalize_coords_batch()