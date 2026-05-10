# AGENTS.md

## Project Context

This repository is a curated Awesome list for trustworthy deep learning papers. The main paper file for Codex-assisted updates is:

- `README.md`: the primary curated paper list shown on GitHub.

Do not edit `FULL_LIST.md` during the default arXiv-paper workflow unless the user explicitly asks for a full-list update. Preserve the current Markdown style, table of contents, badges, image links, and non-paper resource sections unless the user explicitly asks to edit them.

## Default Task: Add an arXiv Paper

When the user gives an arXiv abstract link and asks to use this project's agent instructions, add the paper end-to-end unless it is already present.

### 1. Normalize the arXiv ID

- Accept `https://arxiv.org/abs/<id>`, `http://arxiv.org/abs/<id>`, and arXiv PDF links.
- Canonicalize the list URL to `https://arxiv.org/abs/<base-id>`.
- Strip version suffixes such as `v2` from the list URL unless the user explicitly asks to track a specific version.
- Prefer modern IDs like `2507.16795`; handle legacy IDs only when needed.

### 2. Fetch Metadata Reproducibly

Use the arXiv Atom API as the primary source, not ad hoc scraping of the HTML page. A reliable read-only helper is:

```bash
python3 - "$ARXIV_ID" <<'PY'
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError

raw = sys.argv[1]
match = re.search(r'([a-z-]+/)?\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?', raw)
if not match:
    raise SystemExit(f"Could not find an arXiv id in: {raw}")

paper_id = re.sub(r'v\d+$', '', match.group(0))
url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'id_list': paper_id})
request = urllib.request.Request(
    url,
    headers={'User-Agent': 'awesome-trustworthy-deep-learning-agent/1.0'},
)
for attempt in range(2):
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        break
    except HTTPError as exc:
        if exc.code == 429 and attempt == 0:
            time.sleep(10)
            continue
        raise SystemExit(f"arXiv API request failed with HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise SystemExit(f"arXiv API request failed: {exc.reason}") from exc

ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
root = ET.fromstring(data)
entry = root.find('atom:entry', ns)
if entry is None:
    raise SystemExit(f"No arXiv API entry found for: {paper_id}")

def text(path):
    node = entry.find(path, ns)
    return '' if node is None or node.text is None else ' '.join(node.text.split())

authors = [text_node.text.strip() for text_node in entry.findall('atom:author/atom:name', ns)]
primary = entry.find('arxiv:primary_category', ns)
categories = [node.attrib.get('term', '') for node in entry.findall('atom:category', ns)]

print(json.dumps({
    'id': paper_id,
    'url': f'https://arxiv.org/abs/{paper_id}',
    'title': text('atom:title'),
    'authors': authors,
    'published': text('atom:published'),
    'updated': text('atom:updated'),
    'primary_category': '' if primary is None else primary.attrib.get('term', ''),
    'categories': categories,
    'abstract': text('atom:summary'),
}, indent=2, ensure_ascii=False))
PY
```

If the API is unavailable, use the arXiv abstract page as a fallback and say so in the final response.

### 3. Check for Duplicates

Before editing, search `README.md` for the canonical arXiv ID:

```bash
rg -n "arxiv.org/abs/<ARXIV_ID>|<ARXIV_ID>" README.md
```

If the paper already exists, do not add a duplicate. If the user asked for an update, edit the existing entry in place.

### 4. Choose Exactly One Primary Category

Use the title, abstract, arXiv categories, and nearby existing entries. Prefer the most specific technical category over `Others`. Use `Survey` first when the paper is mainly a survey, SoK, taxonomy, review, or overview.

Allowed paper categories:

- `Survey`: surveys, systematic reviews, SoK papers, taxonomies, tutorials, broad overviews.
- `Out-of-Distribution Generalization`: domain generalization, distribution shift, OOD detection/generalization, open-set/open-world recognition, uncertainty under shift, spurious correlation, robust generalization.
- `Evasion Attacks and Defenses`: inference-time adversarial examples, jailbreaks, prompt injection, red teaming, adversarial robustness, adversarial training, retrieval corruption, defense against evasion.
- `Poisoning Attacks and Defenses`: training-time attacks, data poisoning, backdoors, model poisoning, sleeper agents, poisoned memory or knowledge bases.
- `Privacy`: membership inference, model inversion, data extraction, memorization, differential privacy, privacy leakage, federated privacy, machine unlearning.
- `Fairness`: bias, discrimination, group fairness, individual fairness, equalized performance, fairness in serving or allocation.
- `Interpretability`: explainability, mechanistic interpretability, circuits, sparse autoencoders, feature attribution, concept bottlenecks, probes, representation analysis.
- `Alignment`: RLHF, DPO or preference learning, reward modeling, scalable oversight, constitutional AI, value alignment, alignment failure from fine-tuning, weak-to-strong generalization for oversight.
- `Others`: trustworthy AI topics that do not fit above, including reliability, calibration, hallucination, watermarking, governance, sustainability, blockchain, transparency reports, general AI safety.

If two categories are plausible, pick the one matching the paper's main contribution and include the secondary topic in `Key Word`.

### 5. Generate the Entry

Use this exact structure for new arXiv-only entries:

```markdown
- Paper Title. [[paper]](https://arxiv.org/abs/xxxx.xxxxx)
  - Author One, Author Two, Author Three.
  - Key Word: Keyword One; Keyword Two; Keyword Three.
  - <details><summary>Digest</summary> Concise digest derived from the arXiv abstract.</details>
```

Rules:

- Use the arXiv title with normalized whitespace and a final period before `[[paper]]`.
- Use arXiv author order exactly. End the author line with a period.
- Do not invent conference, journal, project, or code links. Add `[[code]](...)` only when the user supplies it or a reliable official source is found.
- `Key Word` values are inferred controlled tags, not official arXiv metadata. Use 3-6 semicolon-separated phrases.
- This repo stores abstract content as `Digest`. Write a concise 2-4 sentence digest from the abstract instead of pasting a long abstract verbatim, unless the user explicitly asks for verbatim abstract text.
- New entries should close `</details>` on the same line.
- Avoid bolding new paper titles unless the target local section already consistently bolds every entry.

### 6. Insert Into `README.md`

Add accepted arXiv papers to `README.md` only.

- Locate the same `## Category` under `# Paper List`.
- If that category already uses `### YYYY` subsections, insert into the matching year subsection or create one in descending year order.
- If that category does not use year subsections, insert directly after the category's full-list link and before the next paper or heading.
- Sort by initial arXiv `published` date descending. If exact dates are tied, sort by arXiv ID descending.
- Do not remove older existing README entries during a normal add-paper task.

### 7. Validate Before Finishing

Before the final response:

- Re-run the duplicate search for the arXiv ID.
- Inspect `git diff -- README.md AGENTS.md`.
- Confirm that `README.md` was edited, or explain why no edit was needed.

## General Editing Rules

- Keep edits tightly scoped to the requested paper addition or instruction update.
- Preserve existing wording and formatting outside the touched insertion points.
- Use Markdown-compatible ASCII punctuation in new instructions and entries unless the paper title or author names require non-ASCII characters.
- Do not run broad reformatters on these Markdown files.
