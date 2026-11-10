---
name: Fenice
description: A calm, precise research workbench guided by technical wayfinding.
colors:
  bg-window: "#F2F1E9"
  bg-surface: "#FCFCF8"
  bg-panel: "#E7ECE6"
  bg-sidebar: "#173F50"
  bg-sidebar-hover: "#245568"
  sidebar-selected: "#2B6576"
  sidebar-ink: "#F6F6EF"
  sidebar-muted: "#D1E0DD"
  grid: "#D8E1DA"
  grid-major: "#C8D5CD"
  border: "#CED8D1"
  border-strong: "#A7B7AF"
  ink: "#16333D"
  ink-soft: "#3B5961"
  ink-muted: "#506970"
  primary: "#1B5361"
  primary-hover: "#286878"
  primary-pressed: "#153F49"
  primary-tint: "#DCEAE6"
  on-primary: "#FFFFFF"
  on-action: "#FFFFFF"
  action: "#B64E40"
  action-hover: "#A8463A"
  action-pressed: "#963F34"
  route-geometry: "#2F7864"
  route-run: "#8C5B12"
  route-results: "#4D6598"
  module-solver: "#B64E40"
  module-monitor: "#8C5B12"
  module-results: "#4D6598"
  warning: "#83540B"
  error: "#A63C35"
  success: "#286B50"
  disabled-ink: "#899B99"
  disabled-bg: "#E9EDE7"
  console-bg: "#142D37"
  console-ink: "#E6EEE9"
  console-dim: "#A2B7B5"
typography:
  title:
    fontFamily: "Segoe UI, Noto Sans, Inter, sans-serif"
    fontSize: "19px"
    fontWeight: 600
  headline:
    fontFamily: "Segoe UI, Noto Sans, Inter, sans-serif"
    fontSize: "20px"
    fontWeight: 650
  body:
    fontFamily: "Segoe UI, Noto Sans, Inter, sans-serif"
    fontSize: "13px"
    fontWeight: 400
  label:
    fontFamily: "Segoe UI, Noto Sans, Inter, sans-serif"
    fontSize: "12px"
    fontWeight: 400
  overline:
    fontFamily: "Segoe UI, Noto Sans, Inter, sans-serif"
    fontSize: "11px"
    fontWeight: 650
  console:
    fontFamily: "Consolas, monospace"
rounded:
  xs: "4px"
  sm: "6px"
  md: "8px"
  switch-knob: "9px"
spacing:
  micro: "3px"
  fine: "4px"
  compact: "8px"
  regular: "12px"
  roomy: "16px"
  canvas-gutter: "18px"
  wide-control: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "8px 24px"
  button-action:
    backgroundColor: "{colors.action}"
    textColor: "{colors.on-action}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
  button-secondary:
    backgroundColor: "{colors.bg-panel}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
  input:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "5px 9px"
  section-panel:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "16px"
  status-pill:
    backgroundColor: "{colors.primary-tint}"
    textColor: "{colors.primary}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "4px 9px"
  module-item:
    backgroundColor: "transparent"
    textColor: "{colors.sidebar-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "9px 10px"
  graph-node:
    backgroundColor: "{colors.bg-surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    width: "248px"
---

# Design System: Fenice

## Overview

**Creative North Star: "Research Campus Wayfinding"**

Fenice uses the calm legibility of a technical research campus: a marine navigation rail anchors a warm porcelain work surface, while a fixed-size humanist sans keeps dense simulation controls readable at desktop scale. The result is precise and unembellished. It carries the product's technical voice through labeled sections, exact values, and clear state cues rather than decorative branding.

The same route language follows a workflow from geometry through execution to results. Route colors appear on graph ports and links, geometry previews, monitor readouts, and result-folder monitor traces; scalar fields retain their chosen colormap. Postprocessing browses snapshots and monitor curves, while the separate Analysis Tools module reports field statistics and samples line profiles. Selecting a node traces its actual connected dependencies while leaving neighboring branches visible. The graph remains a graph: its topology and labels carry the workflow, and color reinforces them.

**Key Characteristics:**
- Marine rail and warm porcelain surfaces
- Humanist sans with fixed desktop sizing
- Green geometry, ochre execution, and blue results routes
- Native Qt controls with quiet borders and explicit state
- Graph connections follow real dependencies
- Postprocessing and Analysis Tools have separate tasks

## Colors

Porcelain neutrals carry most of the workspace; marine teal marks primary interaction, coral marks run actions, and the route colors identify geometry, execution, and results.

### Primary
- **Deep Marine** (#1B5361): Primary control, selection, and focus color on porcelain surfaces.

### Secondary
- **Coral Action** (#B64E40): Run and add-module actions; it also marks Solver nodes.
- **Geometry Green** (#2F7864): Geometry ports, links, and preview outlines.

### Tertiary
- **Execution Ochre** (#8C5B12): Run ports and links, live monitor traces and progress bars, and the Monitor module accent.
- **Results Blue** (#4D6598): Results ports and links, result-folder monitor traces, analysis line profiles, and the Postprocessing module accent.

### Neutral
- **Warm Porcelain** (#F2F1E9) is the window and canvas ground; **Chalk Porcelain** (#FCFCF8) is used for editable surfaces and cards; **Sage Porcelain** (#E7ECE6) separates panels and read-only fields.
- **Marine Rail** (#173F50) anchors the persistent module navigation; **Marine Hover** (#245568) and **Rail Selection** (#2B6576) distinguish hover and selection, while **Rail Ink** (#F6F6EF) and **Rail Muted** (#D1E0DD) carry its title and supporting text.
- **Deep Ink** (#16333D) is the main text color; **Soft Ink** (#3B5961) supports secondary labels; **Muted Ink** (#506970) is used for hints and subtitles.
- **Hairline** (#CED8D1), **Strong Hairline** (#A7B7AF), **Grid** (#D8E1DA), and **Major Grid** (#C8D5CD) divide controls and structure the canvas.
- **Console Night** (#142D37), **Console Text** (#E6EEE9), and **Console Dim** (#A2B7B5) keep solver output distinct from editable forms.
- Disabled text and surfaces use **Disabled Ink** (#899B99) and **Disabled Sage** (#E9EDE7); semantic feedback uses **Warning Ochre** (#83540B), **Error Red** (#A63C35), and **Success Green** (#286B50).

**The Route Key Rule.** Route hue identifies port type, not run state. Keep the port label, status text, or legend beside every color cue, and reserve selection emphasis for the connected path.

## Typography

Segoe UI is the preferred desktop face, with Noto Sans and Inter as fallbacks. The base UI is 13px; role sizes are deliberately close so labels remain readable without turning technical screens into editorial layouts.

- **Title** (19px, weight 600): Window and dialog title.
- **Headline** (20px, weight 650): Empty-workbench prompt.
- **Body** (13px, weight 400): Form values, node labels, and ordinary interface text.
- **Label** (12px, weight 400): Supporting copy, subtitles, and field notes.
- **Overline** (11px, weight 650): Compact section labels.
- **Console** (Consolas, monospace): Solver output and log text.

**The Fixed-Size Wayfinding Rule.** Keep the interface type at its established desktop scale; use weight, spacing, and ink contrast to create hierarchy rather than shrinking labels or adding oversized display type.

## Layout

The workbench keeps a fixed-width module rail beside a flexible graph canvas. The canvas has a compact workflow header and a large open drawing area; the run and monitor region stays quiet until a Monitor is connected. On its first reveal, the lower Monitor region opens to about 34% of the splitter height (at least 220px, at most half). Keep the user's divider position when switching or adding Monitor tabs while that region remains open. At the 1120px minimum window width, preserve the rail and graph as the primary split.

Solver setup uses a west-side top-level tab stack with nested section tabs, grouped fields, and scrollable pages. The current nested labels are “Case and physics,” “Domain and boundaries,” and “Numerics and output.” The Geometry editor pairs a constrained control column with an expanding preview. Postprocessing and Analysis Tools are separate result consumers: the first browses snapshots and monitor curves; the second measures fields and samples profiles. All surfaces share the same palette and type system.

Use a small, repeated spacing vocabulary: 3–4px for compact controls, 8px for local padding, 12px for related groups, and 16–18px for panel and canvas insets. Keep controls and section groups aligned to form rows; allow the graph and preview to absorb available space.

**The Graph Truth Rule.** Show the actual links and branches. Selecting a node highlights its upstream dependencies and downstream work; keep other branches present and visibly subdued rather than rearranging the graph into a fixed sequence.

## Elevation & Depth

Depth comes from tonal layering, fine borders, and the graph's spatial arrangement. The workbench and form surfaces do not rely on drop shadows. The dark navigation rail and solver console create the strongest contrast; cards and controls remain close to the porcelain ground.

## Shapes

Use softly rounded rectangles, with tighter corners on controls and slightly roomier corners on panels and graph nodes. Inputs, buttons, module items, tabs, and group panels share bordered, restrained shapes. The graph nodes use a thin neutral outline, a short type-colored top rule, and a stronger marine outline when selected. Ports are circular and links are curved.

## Components

### Buttons
- **Shape:** Gently rounded corners (6px).
- **Primary:** Marine fill with white text; used for direct form actions such as applying a mesh.
- **Action:** Coral fill with white action text for starting a run or adding the selected module.
- **Secondary:** Sage porcelain fill with a hairline; hover shifts toward marine tint.
- **States:** Hover deepens or tints the control; pressed states use a darker fill. Disabled controls use the disabled ink and surface colors.

### Inputs / Fields
- **Style:** Porcelain fill, strong hairline, 6px radius, and compact horizontal padding (5px 9px).
- **Focus:** A 2px marine border with compensating padding keeps fields from shifting.
- **Read-only / Disabled:** Read-only fields move to sage porcelain; disabled fields use the subdued disabled pair.

### Cards / Containers
- **Corner Style:** 8px on group panels and graph nodes.
- **Background:** Chalk porcelain over the warm porcelain work surface.
- **Border:** Fine neutral hairline; no shadow.
- **Internal Padding:** 16px for group panels, with tighter 6–12px insets inside graph nodes.

### Chips
- **Style:** The status pill uses a pale marine tint, marine text, an 8px radius, and compact padding.
- **State:** Route type is shown by colored ports and links alongside text labels; route colors are not used as unlabeled status chips.

### Navigation
- **Workbench:** A persistent marine rail contains the module palette, add action, run state, and documentation links. Items use pale text and a soft marine selected/hover surface.
- **Setup:** Native tab navigation separates Configuration, Immersed body, Hot start, and Adaptive mesh. Nested sections are “Case and physics,” “Domain and boundaries,” and “Numerics and output.”

### Graph Nodes and Links
- Nodes are light cards with a 4px type-colored top accent, explicit title and status, and labeled input/output ports. Geometry, Solver, Monitor, and Postprocessing use green, coral, ochre, and blue accents; Analysis Tools uses marine primary.
- Typed ports are circles filled when linked; matching curved links inherit the port type color. Selection strengthens the connected route. Unrelated links remain visibly subdued at about 55% opacity rather than disappearing.

### Geometry Preview
- The domain and mesh sit on a light plot surface. The initial view shows exact simulation bounds from x = 0 to Lx and y = 0 to Ly, with equal axis scaling so the domain is not distorted. Grid lines use subdued sage hairlines; boundary labels and cell fills remain legible against the canvas. Immersed-body outlines use the geometry route green. Mesh controls stay together in the left column.

### Monitor and Results
- Monitor separates Plots and Console. The time-step, wave-speed, and monitor-quantity traces use execution ochre; residual quantities retain distinct labeled ink colors. The residual plot receives the largest share of the plot stack; progress and textual state accompany the plots.
- The Console uses a dark marine surface and monospace text. Postprocessing follows the linked run's snapshots and monitor data; its monitor-file traces use results blue while scalar fields keep their selected colormap.

### Analysis Tools
- This is a separate result consumer, not another snapshot viewer. It reports minimum, maximum, mean, and standard deviation for the displayed field or slice, and samples a selected field along a coordinate-defined line.
- The plotted line profile uses results blue and can be exported as CSV. Keep the field, snapshot, slice, statistics, and profile controls together beside the expanding plot.

**The Native Control Rule.** Keep the controls and navigation recognizable as Qt widgets. Use the shared palette and restrained state styling to make them belong to Fenice.

## Do's and Don'ts

### Do:
- **Do** use porcelain surfaces and the marine rail to keep the primary workspace calm and scan-friendly.
- **Do** preserve the geometry green, execution ochre, and results blue route mapping across graph links and readouts.
- **Do** pair color with port labels, legends, and textual run status.
- **Do** retain native Qt navigation and controls, with visible selected and focused states.
- **Do** let the central graph and geometry preview expand while keeping their control areas aligned.

### Don't:
- **Don't** imply a mandatory linear workflow when the graph supports shared geometry and sibling solver branches.
- **Don't** turn the product into a literal campus sign, transit map, timetable, or decorative route diagram.
- **Don't** rely on route color alone to communicate type, progress, or validation.
- **Don't** add shadows or decorative badges that compete with the graph and scientific readouts.
- **Don't** shrink technical labels below the established desktop type scale.
