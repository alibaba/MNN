#!/usr/bin/env python3
import argparse
import re
import sys
import xml.etree.ElementTree as ET


def parse_center(bounds: str):
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds or "")
    if not m:
        return None
    x1, y1, x2, y2 = map(int, m.groups())
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def match(node, rid, text, clazz, contains_text):
    if rid and node.attrib.get("resource-id") != rid:
        return False
    if text and node.attrib.get("text") != text:
        return False
    if clazz and node.attrib.get("class") != clazz:
        return False
    if contains_text and contains_text not in (node.attrib.get("text") or ""):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--resource-id")
    ap.add_argument("--text")
    ap.add_argument("--class-name")
    ap.add_argument("--contains-text")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    root = ET.parse(args.xml).getroot()
    hits = []
    for node in root.iter("node"):
        if match(node, args.resource_id, args.text, args.class_name, args.contains_text):
            center = parse_center(node.attrib.get("bounds", ""))
            if center:
                hits.append((center[0], center[1], node.attrib.get("bounds", "")))

    if not hits:
        print("NOT_FOUND", file=sys.stderr)
        sys.exit(2)

    idx = max(0, min(args.index, len(hits) - 1))
    x, y, bounds = hits[idx]
    print(f"{x} {y} {bounds}")


if __name__ == "__main__":
    main()
