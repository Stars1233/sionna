#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Add cross-document backlinks to bibliography entries.

sphinxcontrib-bibtex only supports backrefs within the same document.
Since our bibliographies live on dedicated references pages while all
citations are in API docs and tutorials, this extension fills the gap
by appending explicit backlinks to each bibliography entry.
"""

from collections import defaultdict
from typing import cast

import docutils.nodes
from sphinx.application import Sphinx


def _add_backrefs(app: Sphinx, doctree, docname: str):
    """Append cross-document backlinks to every citation node."""
    domain = cast("BibtexDomain", app.env.get_domain("cite"))
    builder = app.builder

    # Build a map: bibtex_key -> [(docname, anchor_id), ...]
    key_to_refs = defaultdict(list)
    for cref in domain.citation_refs:
        for target in cref.targets:
            key_to_refs[target.key].append(
                (cref.docname, cref.citation_ref_id)
            )

    for citation_node in doctree.findall(docutils.nodes.citation):
        # The label text equals the BibTeX key (via our KeyLabelStyle)
        label_node = citation_node.next_node(docutils.nodes.label)
        if label_node is None:
            continue
        key = label_node.astext()

        refs = key_to_refs.get(key, [])
        if not refs:
            continue

        # De-duplicate by (docname, anchor), then sort for deterministic output
        # across parallel builds (citation_refs iteration order is unstable).
        seen = set()
        unique_refs = []
        for ref_docname, ref_id in refs:
            pair = (ref_docname, ref_id)
            if pair not in seen:
                seen.add(pair)
                unique_refs.append(pair)
        unique_refs.sort()

        # Build URI list
        links = []
        for ref_docname, ref_id in unique_refs:
            try:
                uri = (
                    builder.get_relative_uri(docname, ref_docname)
                    + "#"
                    + ref_id
                )
                links.append(uri)
            except Exception:
                pass

        if not links:
            continue

        # Also set citation_node["backrefs"] so themes that render
        # them (e.g. making the label clickable) can benefit when
        # there is exactly one backref.  For cross-doc this won't
        # work in vanilla docutils, so we add explicit nodes below.

        # Append a small paragraph with numbered back-links
        para = docutils.nodes.paragraph(classes=["cite-backrefs"])
        if len(links) == 1:
            ref = docutils.nodes.reference(
                "", "\u21a9", refuri=links[0], classes=["cite-backref"]
            )
            para += ref
        else:
            para += docutils.nodes.Text("\u21a9 ")
            for i, uri in enumerate(links):
                if i > 0:
                    para += docutils.nodes.Text(" ")
                ref = docutils.nodes.reference(
                    "",
                    str(i + 1),
                    refuri=uri,
                    classes=["cite-backref"],
                )
                para += ref

        citation_node += para


def setup(app: Sphinx):
    app.connect("doctree-resolved", _add_backrefs)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
