"""A small fictional corpus about a robotics ecosystem.

Every fact is a short sentence so that the deterministic rule-based extractor can build a
knowledge graph from it without an LLM. All people and companies are invented.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


DOCS: list[Document] = [
    Document(
        "d01",
        "Helios Robotics",
        "Helios Robotics is a robotics company. Maria Chen founded Helios Robotics. "
        "Helios Robotics is headquartered in Austin. Helios Robotics manufactures Atlas Arm. "
        "Helios Robotics manufactures Orbit Rover.",
    ),
    Document(
        "d02",
        "Maria Chen",
        "Maria Chen is a roboticist and entrepreneur. Maria Chen studied at Stanford University. "
        "Maria Chen founded Helios Robotics after years of research on manipulators.",
    ),
    Document(
        "d03",
        "Nimbus Vision",
        "Nimbus Vision builds machine vision software. Tomas Ruiz founded Nimbus Vision. "
        "Nimbus Vision is located in Boston. Helios Robotics acquired Nimbus Vision.",
    ),
    Document(
        "d04",
        "Tomas Ruiz",
        "Tomas Ruiz is a computer vision researcher. "
        "Tomas Ruiz studied at Massachusetts Institute of Technology. "
        "Tomas Ruiz founded Nimbus Vision.",
    ),
    Document(
        "d05",
        "Voltix Batteries",
        "Voltix Batteries makes battery cells for robots. Anika Rao founded Voltix Batteries. "
        "Voltix Batteries is located in Reno. Voltix Batteries supplies Helios Robotics. "
        "Voltix Batteries manufactures Voltix Cells.",
    ),
    Document(
        "d06",
        "Anika Rao",
        "Anika Rao is an electrochemist. Anika Rao studied at Carnegie Mellon University. "
        "Anika Rao leads Voltix Batteries.",
    ),
    Document(
        "d07",
        "Atlas Arm",
        "Atlas Arm is a six-axis industrial manipulator. Atlas Arm uses Voltix Cells. "
        "Helios Robotics manufactures Atlas Arm.",
    ),
    Document(
        "d08",
        "Orbit Rover",
        "Orbit Rover is a planetary exploration rover. Orbit Rover uses Quanta Chip. "
        "Mars Survey Institute uses Orbit Rover. Helios Robotics manufactures Orbit Rover.",
    ),
    Document(
        "d09",
        "Mars Survey Institute",
        "Mars Survey Institute studies the surface of Mars. "
        "Mars Survey Institute is located in Pasadena. "
        "Mars Survey Institute partnered with Northwind Labs.",
    ),
    Document(
        "d10",
        "Northwind Labs",
        "Northwind Labs designs AI accelerator chips. Daniel Okafor founded Northwind Labs. "
        "Northwind Labs is headquartered in Seattle. Northwind Labs manufactures Quanta Chip. "
        "Northwind Labs partnered with Helios Robotics.",
    ),
    Document(
        "d11",
        "Daniel Okafor",
        "Daniel Okafor is a chip architect. Daniel Okafor studied at University of Washington. "
        "Daniel Okafor founded Northwind Labs.",
    ),
    Document(
        "d12",
        "Greenfield Capital",
        "Greenfield Capital is a venture capital firm. "
        "Greenfield Capital is headquartered in New York. "
        "Greenfield Capital invested in Helios Robotics. "
        "Greenfield Capital invested in Voltix Batteries.",
    ),
    Document(
        "d13",
        "Sofia Marin",
        "Sofia Marin is a partner at the fund. Sofia Marin works at Greenfield Capital. "
        "Sofia Marin studied at Columbia University.",
    ),
    Document(
        "d14",
        "Quanta Chip",
        "Quanta Chip is an AI accelerator for edge devices. It draws under ten watts.",
    ),
    Document(
        "d15",
        "Industry notes",
        "Robotics adoption is growing in logistics, manufacturing and space exploration. "
        "Battery supply and AI chips are the main bottlenecks for robot makers. "
        "Analysts expect more acquisitions of vision startups.",
    ),
]

DOC_BY_ID = {d.id: d for d in DOCS}
