#!/usr/bin/env python3
"""
HMARL-SOC Manuscript AI Detection & Humanization Automation
Uses Undetectable.ai APIs to:
  1. Extract prose sections from LaTeX
  2. Detect AI-written content per section
  3. Optionally humanize flagged sections
  4. Write back humanized text to LaTeX

Usage:
  # Check credits
  python ai_check.py --api-key YOUR_KEY --check-credits

  # Detect only (no humanization, no credits used)
  python ai_check.py --api-key YOUR_KEY --detect

  # Detect + humanize flagged sections (uses credits)
  python ai_check.py --api-key YOUR_KEY --humanize

  # Humanize a specific section only
  python ai_check.py --api-key YOUR_KEY --humanize --section "Introduction"
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ── API endpoints ──
DETECT_URL = "https://ai-detect.undetectable.ai"
HUMANIZE_URL = "https://humanize.undetectable.ai"

# ── LaTeX section extraction ──
SECTION_PATTERN = re.compile(
    r'\\section\*?\{([^}]+)\}',
    re.MULTILINE
)

SUBSECTION_PATTERN = re.compile(
    r'\\subsection\*?\{([^}]+)\}',
    re.MULTILINE
)


def strip_latex_commands(text: str) -> str:
    """Remove LaTeX commands, keeping readable text for AI detection."""
    # Remove comments
    text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
    # Remove begin/end environments for math, algorithm, table, figure, tikz
    text = re.sub(r'\\begin\{(equation|align|algorithm|algorithmic|table|tabular|figure|tikzpicture)\*?\}.*?\\end\{\1\*?\}', '', text, flags=re.DOTALL)
    # Remove \cite, \ref, \label commands
    text = re.sub(r'\\(?:cite|ref|label|eqref)\{[^}]*\}', '', text)
    # Remove \textbf, \textit, \emph — keep content
    text = re.sub(r'\\(?:textbf|textit|emph|textsc|textrm)\{([^}]*)\}', r'\1', text)
    # Remove inline math $...$
    text = re.sub(r'\$[^$]+\$', '', text)
    # Remove \item
    text = re.sub(r'\\item\s*', '- ', text)
    # Remove remaining \commands
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*', '', text)
    # Remove braces
    text = re.sub(r'[{}]', '', text)
    # Remove ~
    text = text.replace('~', ' ')
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def extract_sections(tex_path: str) -> list[dict]:
    """Extract prose sections from LaTeX file."""
    content = Path(tex_path).read_text(encoding='utf-8')
    
    # Find all \section positions
    sections = []
    for m in SECTION_PATTERN.finditer(content):
        sections.append({
            'name': m.group(1).strip(),
            'start': m.start(),
            'type': 'section'
        })
    
    # Add end marker
    bib_match = re.search(r'\\begin\{thebibliography\}', content)
    end_pos = bib_match.start() if bib_match else len(content)
    
    # Extract text between sections
    result = []
    for i, sec in enumerate(sections):
        next_start = sections[i + 1]['start'] if i + 1 < len(sections) else end_pos
        raw_text = content[sec['start']:next_start]
        clean_text = strip_latex_commands(raw_text)
        
        word_count = len(clean_text.split())
        if word_count < 50:  # Skip very short sections
            continue
            
        result.append({
            'name': sec['name'],
            'raw_latex': raw_text,
            'clean_text': clean_text,
            'word_count': word_count,
            'start_pos': sec['start'],
            'end_pos': next_start,
        })
    
    return result


def check_credits(api_key: str) -> dict:
    """Check remaining word credits."""
    resp = requests.get(
        f"{DETECT_URL}/check-user-credits",
        headers={"apikey": api_key}
    )
    resp.raise_for_status()
    return resp.json()


def detect_ai(text: str, api_key: str) -> dict:
    """Submit text for AI detection and poll until complete."""
    # Submit
    payload = {
        "text": text.replace('\n', '\\n'),
        "key": api_key,
        "document_type": "Document",
        "model": "xlm_ud_detector",
    }
    
    resp = requests.post(f"{DETECT_URL}/detect", json=payload)
    resp.raise_for_status()
    data = resp.json()
    
    if "error" in data:
        return {"error": data["error"]}
    
    doc_id = data["id"]
    print(f"    Submitted (ID: {doc_id[:8]}...) — polling for results", end="", flush=True)
    
    # Poll until done
    for attempt in range(30):
        time.sleep(3)
        print(".", end="", flush=True)
        
        query_resp = requests.post(
            f"{DETECT_URL}/query",
            json={"id": doc_id}
        )
        query_resp.raise_for_status()
        result = query_resp.json()
        
        if result and result.get("status") not in ("pending", "analyzing"):
            print(" done!")
            return result
    
    print(" timeout!")
    return {"error": "Polling timeout", "last_status": result}


def humanize_text(text: str, api_key: str,
                  readability: str = "University",
                  purpose: str = "Academic",
                  strength: str = "More Human") -> dict:
    """Submit text for humanization and poll until complete."""
    payload = {
        "content": text,
        "readability": readability,
        "purpose": purpose,
        "strength": strength,
        "model": "v2",
        "document_type": "Text",
    }
    
    resp = requests.post(
        f"{HUMANIZE_URL}/submit",
        json=payload,
        headers={"apikey": api_key}
    )
    resp.raise_for_status()
    data = resp.json()
    
    if "error" in data:
        return {"error": data["error"]}
    
    doc_id = data.get("id")
    if not doc_id:
        return {"error": "No document ID returned", "response": data}
    
    print(f"    Humanizing (ID: {doc_id[:8]}...) — polling", end="", flush=True)
    
    # Poll until done
    for attempt in range(60):
        time.sleep(5)
        print(".", end="", flush=True)
        
        doc_resp = requests.post(
            f"{HUMANIZE_URL}/document",
            json={"id": doc_id},
            headers={"apikey": api_key}
        )
        doc_resp.raise_for_status()
        result = doc_resp.json()
        
        status = result.get("status", "")
        if status == "done" or result.get("output"):
            print(" done!")
            return result
        elif "error" in result:
            print(f" error!")
            return result
    
    print(" timeout!")
    return {"error": "Polling timeout"}


def format_detection_result(result: dict) -> str:
    """Format detection result for display."""
    if "error" in result:
        return f"  ❌ Error: {result['error']}"
    
    score = result.get("result")
    label = result.get("label", "Unknown")
    details = result.get("result_details", {})
    
    lines = []
    
    # Overall score
    if score is not None:
        emoji = "🟢" if score < 50 else ("🟡" if score < 70 else "🔴")
        lines.append(f"  {emoji} Overall: {score:.0f}/100 ({label})")
    
    # Detector breakdown
    if details:
        detector_names = {
            "scoreGptZero": "GPTZero",
            "scoreOpenAI": "OpenAI",
            "scoreWriter": "Writer",
            "scoreCrossPlag": "CrossPlag",
            "scoreCopyLeaks": "CopyLeaks",
            "scoreSapling": "Sapling",
            "scoreContentAtScale": "ContentAtScale",
            "scoreZeroGPT": "ZeroGPT",
            "human": "Human Score",
        }
        lines.append("  Detector Scores:")
        for key, name in detector_names.items():
            val = details.get(key)
            if val is not None:
                emoji = "🟢" if val < 50 else ("🟡" if val < 70 else "🔴")
                lines.append(f"    {emoji} {name}: {val:.0f}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AI Detection & Humanization for LaTeX manuscripts")
    parser.add_argument("--api-key", required=True, help="Undetectable.ai API key")
    parser.add_argument("--tex-file", default="main.tex", help="LaTeX file path (default: main.tex)")
    parser.add_argument("--check-credits", action="store_true", help="Check remaining credits")
    parser.add_argument("--detect", action="store_true", help="Run AI detection on all sections")
    parser.add_argument("--humanize", action="store_true", help="Detect + humanize flagged sections")
    parser.add_argument("--section", type=str, help="Only process this section (by name)")
    parser.add_argument("--threshold", type=float, default=50, help="AI score threshold to flag (default: 50)")
    parser.add_argument("--readability", default="University", help="Readability level")
    parser.add_argument("--purpose", default="Academic", help="Purpose (Academic, General, etc.)")
    parser.add_argument("--strength", default="More Human", help="Humanization strength")
    
    args = parser.parse_args()
    
    # ── Check credits ──
    if args.check_credits:
        print("💰 Checking credits...")
        credits = check_credits(args.api_key)
        print(f"  Base credits:  {credits.get('baseCredits', '?')}")
        print(f"  Boost credits: {credits.get('boostCredits', '?')}")
        print(f"  Total credits: {credits.get('credits', '?')}")
        if not (args.detect or args.humanize):
            return
    
    if not (args.detect or args.humanize):
        parser.print_help()
        return
    
    # ── Extract sections ──
    tex_path = Path(args.tex_file)
    if not tex_path.exists():
        print(f"❌ File not found: {tex_path}")
        sys.exit(1)
    
    print(f"\n📄 Extracting sections from {tex_path}...")
    sections = extract_sections(str(tex_path))
    
    if args.section:
        sections = [s for s in sections if args.section.lower() in s['name'].lower()]
        if not sections:
            print(f"❌ Section '{args.section}' not found")
            sys.exit(1)
    
    print(f"  Found {len(sections)} sections:")
    for s in sections:
        print(f"    • {s['name']} ({s['word_count']} words)")
    
    total_words = sum(s['word_count'] for s in sections)
    print(f"  Total: {total_words} words")
    
    # ── Detect AI ──
    print(f"\n🔍 Running AI detection (threshold: {args.threshold})...\n")
    flagged = []
    
    for sec in sections:
        if sec['word_count'] < 200:
            print(f"⏭️  {sec['name']} — skipped (only {sec['word_count']} words, need 200+)")
            continue
        
        print(f"📝 {sec['name']} ({sec['word_count']} words):")
        result = detect_ai(sec['clean_text'], args.api_key)
        print(format_detection_result(result))
        
        score = result.get("result", 0)
        if score and score >= args.threshold:
            flagged.append({**sec, 'detection_result': result})
        
        print()
        time.sleep(1)  # Rate limit
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("📊 DETECTION SUMMARY")
    print("=" * 60)
    print(f"  Sections scanned: {len(sections)}")
    print(f"  Sections flagged: {len(flagged)}")
    if flagged:
        print(f"  Flagged sections:")
        for f in flagged:
            score = f['detection_result'].get('result', '?')
            print(f"    🔴 {f['name']} (score: {score})")
    else:
        print("  ✅ No sections flagged as AI-written!")
    
    # ── Humanize ──
    if args.humanize and flagged:
        print(f"\n🤖 Humanizing {len(flagged)} flagged sections...")
        humanized_count = 0
        
        for sec in flagged:
            print(f"\n📝 Humanizing: {sec['name']}...")
            result = humanize_text(
                sec['clean_text'], 
                args.api_key,
                readability=args.readability,
                purpose=args.purpose,
                strength=args.strength,
            )
            
            if "error" in result:
                print(f"  ❌ Error: {result['error']}")
                continue
            
            output = result.get("output", "")
            if output:
                # Save humanized text to a file for manual review
                out_path = tex_path.parent / f"humanized_{sec['name'].replace(' ', '_').lower()}.txt"
                out_path.write_text(output, encoding='utf-8')
                print(f"  ✅ Saved humanized text to: {out_path}")
                humanized_count += 1
            else:
                print(f"  ⚠️  No output received")
                print(f"  Full response: {json.dumps(result, indent=2)[:500]}")
        
        print(f"\n✅ Humanized {humanized_count}/{len(flagged)} sections")
        print("⚠️  Review the humanized_*.txt files before applying to LaTeX!")
        print("   Technical terms may have been altered — verify carefully.")
    
    elif args.humanize and not flagged:
        print("\n✅ No sections to humanize — all passed!")


if __name__ == "__main__":
    main()
