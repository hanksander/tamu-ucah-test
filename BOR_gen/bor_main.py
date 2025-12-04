# Option 1: Custom function
from body_of_revolution_mesh import generate_bor_mesh, write_tri_file

def my_radius(x, length, width):
    return (width/length) * x  # Your cone example

vertices, triangles, stats = generate_bor_mesh(
    r_func=my_radius,
    x_range=(0, 1),
    n_axial=100,
    n_circumferential=60,
    r_func_args={'length': 1, 'width': 0.2}
)

write_tri_file("my_cone.tri", vertices, triangles)

# Option 2: Built-in shape
from bor_shape_library import VonKarmanNose

shape = VonKarmanNose(length=10, base_radius=1)
vertices, triangles, stats = shape.generate_mesh()
write_tri_file("vonkarman.tri", vertices, triangles)

# Option 3: Generate family
from bor_shape_library import generate_shape_family, TangentOgive

generate_shape_family(
    TangentOgive,
    length_range=(1, 1),
    radius_range=(0.2, 0.2),
    n_lengths=1,
    n_radii=1
)
# Creates 1 VK