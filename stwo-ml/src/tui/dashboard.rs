//! ObelyZK Proof Dashboard — terminal visualization of verifiable ML inference.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Padding, Paragraph, Wrap},
    Frame,
};

// ── ObelyZK Brand Colors ────────────────────────────────────────────
const BRAND:      Color = Color::Rgb(163, 230, 53);   // Lime green
const BRAND_DIM:  Color = Color::Rgb(101, 163, 13);   // Dark lime
const ACCENT:     Color = Color::Rgb(52, 211, 153);    // Emerald
const SURFACE:    Color = Color::Rgb(18, 18, 26);      // Dark surface
const TEXT:       Color = Color::Rgb(244, 244, 245);   // Zinc 100
const TEXT_DIM:   Color = Color::Rgb(113, 113, 122);   // Zinc 500
const TEXT_MUTED: Color = Color::Rgb(63, 63, 70);      // Zinc 700
const DANGER:     Color = Color::Rgb(239, 68, 68);     // Red
const SUCCESS:    Color = Color::Rgb(34, 197, 94);     // Green
const WARN:       Color = Color::Rgb(250, 204, 21);    // Yellow
const HASH_COLOR: Color = Color::Rgb(139, 92, 246);    // Violet

/// Dashboard state.
#[derive(Debug, Clone)]
pub struct DashboardState {
    pub model_name: String,
    pub model_params: String,
    pub model_layers: u32,
    pub num_turns: usize,
    pub tokens_in: usize,
    pub tokens_out: usize,

    pub step: PipelineStep,
    pub capture_progress: f64,
    pub prove_progress: f64,
    pub recursive_progress: f64,
    pub verify_progress: f64,

    pub capture_time: Option<f64>,
    pub prove_time: Option<f64>,
    pub recursive_time: Option<f64>,

    pub weight_commitment: Option<String>,
    pub io_root: Option<String>,
    pub report_hash: Option<String>,

    pub contract: String,
    pub network: String,
    pub verification_count: Option<u64>,

    pub tamper_io: Option<bool>,
    pub tamper_weight: Option<bool>,
    pub tamper_output: Option<bool>,

    pub turns: Vec<(String, String)>,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStep {
    Idle,
    Capture,
    Prove,
    Recursive,
    Verify,
    Complete,
}

impl PipelineStep {
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Idle => 0, Self::Capture => 1, Self::Prove => 2,
            Self::Recursive => 3, Self::Verify => 4, Self::Complete => 5,
        }
    }
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            model_name: "qwen2-0.5b".into(),
            model_params: "247,726,080".into(),
            model_layers: 169,
            num_turns: 0, tokens_in: 0, tokens_out: 0,
            step: PipelineStep::Idle,
            capture_progress: 0.0, prove_progress: 0.0,
            recursive_progress: 0.0, verify_progress: 0.0,
            capture_time: None, prove_time: None, recursive_time: None,
            weight_commitment: None, io_root: None, report_hash: None,
            contract: "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005".into(),
            network: "Starknet Sepolia".into(),
            verification_count: None,
            tamper_io: None, tamper_weight: None, tamper_output: None,
            turns: Vec::new(), logs: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Render
// ═══════════════════════════════════════════════════════════════════════

pub fn render(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();

    // Clear background
    frame.render_widget(Block::default().style(Style::default().bg(SURFACE)), area);

    let main = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // Header
            Constraint::Min(12),   // Body
            Constraint::Length(3), // Footer
        ])
        .split(area);

    render_header(frame, main[0], state);
    render_body(frame, main[1], state);
    render_footer(frame, main[2], state);
}

fn render_header(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let logo = vec![
        Line::from(vec![
            Span::styled("  ▗▄▖ ▗▄▄▖ ▗▄▄▄▖▗▖  ▗▖ ▗▖ ▗▖▗▄▄▄▖▗▖ ▗▖", Style::default().fg(BRAND)),
        ]),
        Line::from(vec![
            Span::styled(" ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌  ▝▜▌▐▛▘   ▄▄▄▘▐▌▗▞▘", Style::default().fg(BRAND)),
            Span::raw("    "),
            Span::styled("Verifiable ML Inference", Style::default().fg(TEXT).bold()),
        ]),
        Line::from(vec![
            Span::styled(" ▐▌ ▐▌▐▛▀▚▖▐▛▀▀▘▐▌   ▐▌▐▌  ▗▄▄▄▖ ▐▛▚▖", Style::default().fg(BRAND_DIM)),
            Span::raw("    "),
            Span::styled(&state.model_name, Style::default().fg(ACCENT)),
            Span::styled(
                format!(" • {} params • {} layers", state.model_params, state.model_layers),
                Style::default().fg(TEXT_DIM),
            ),
        ]),
        Line::from(vec![
            Span::styled(" ▝▚▄▞▘▐▙▄▞▘▐▙▄▄▖▐▙▄▄▖▐▌▐▌  ▐▌  ▐▌▐▌ ▐▌", Style::default().fg(BRAND_DIM)),
            Span::raw("    "),
            Span::styled("Powered by STWO", Style::default().fg(TEXT_MUTED)),
        ]),
    ];

    let header = Paragraph::new(logo)
        .block(Block::default().style(Style::default().bg(SURFACE)));
    frame.render_widget(header, area);
}

fn render_body(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    render_left(frame, body[0], state);
    render_right(frame, body[1], state);
}

fn render_left(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(14), // Pipeline
            Constraint::Min(4),    // Logs
        ])
        .split(area);

    // Pipeline block
    let pipeline_block = Block::default()
        .title(Span::styled(" Proving Pipeline ", Style::default().fg(BRAND).bold()))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BRAND_DIM))
        .style(Style::default().bg(SURFACE))
        .padding(Padding::new(1, 1, 1, 0));

    let pipeline_inner = pipeline_block.inner(layout[0]);
    frame.render_widget(pipeline_block, layout[0]);

    let steps = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(1), // spacer
            Constraint::Min(1),   // coverage
        ])
        .split(pipeline_inner);

    render_pipeline_step(frame, steps[0], "1", "Capture",
        "M31 forward passes (24 layers)", state.capture_progress,
        state.capture_time, state.step.as_u8() >= PipelineStep::Capture.as_u8());

    render_pipeline_step(frame, steps[1], "2", "GKR Prove",
        "96 matmul sumchecks", state.prove_progress,
        state.prove_time, state.step.as_u8() >= PipelineStep::Prove.as_u8());

    render_pipeline_step(frame, steps[2], "3", "Recursive",
        "STARK compression → 1 TX", state.recursive_progress,
        state.recursive_time, state.step.as_u8() >= PipelineStep::Recursive.as_u8());

    render_pipeline_step(frame, steps[3], "4", "Verify",
        "Self-verify + on-chain", state.verify_progress,
        None, state.step.as_u8() >= PipelineStep::Verify.as_u8());

    // Coverage summary
    let coverage = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("  96", Style::default().fg(BRAND).bold()),
            Span::styled(" MatMul  ", Style::default().fg(TEXT_DIM)),
            Span::styled("24", Style::default().fg(BRAND).bold()),
            Span::styled(" SiLU  ", Style::default().fg(TEXT_DIM)),
            Span::styled("49", Style::default().fg(BRAND).bold()),
            Span::styled(" RMSNorm", Style::default().fg(TEXT_DIM)),
        ]),
    ]);
    frame.render_widget(coverage, steps[5]);

    // Logs
    let log_items: Vec<ListItem> = state.logs.iter().rev().take(6).map(|l| {
        ListItem::new(Span::styled(l.clone(), Style::default().fg(TEXT_MUTED)))
    }).collect();
    let logs = List::new(log_items)
        .block(Block::default()
            .title(Span::styled(" Log ", Style::default().fg(TEXT_DIM)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(TEXT_MUTED))
            .style(Style::default().bg(SURFACE)));
    frame.render_widget(logs, layout[1]);
}

fn render_pipeline_step(
    frame: &mut Frame, area: Rect,
    num: &str, name: &str, desc: &str,
    progress: f64, time: Option<f64>, active: bool,
) {
    let (color, icon) = if progress >= 1.0 {
        (SUCCESS, "✓")
    } else if active && progress > 0.0 {
        (BRAND, "▸")
    } else {
        (TEXT_MUTED, "·")
    };

    let time_str = time.map(|t| format!(" {:.1}s", t)).unwrap_or_default();

    let label = Line::from(vec![
        Span::styled(format!(" {icon} "), Style::default().fg(color)),
        Span::styled(format!("[{num}] "), Style::default().fg(TEXT_DIM)),
        Span::styled(name, Style::default().fg(TEXT).bold()),
        Span::styled(format!("  {desc}"), Style::default().fg(TEXT_DIM)),
        Span::styled(time_str, Style::default().fg(ACCENT)),
    ]);

    frame.render_widget(Paragraph::new(label), area);

    // Progress bar on second line
    if area.height > 1 {
        let bar_area = Rect { y: area.y + 1, height: 1, ..area };
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 30, 40)))
            .ratio(progress.clamp(0.0, 1.0))
            .label("");
        frame.render_widget(gauge, bar_area);
    }
}

fn render_right(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10), // Commitments + on-chain
            Constraint::Length(7),  // Tamper
            Constraint::Min(4),    // Conversation
        ])
        .split(area);

    // Commitments
    let mut lines = vec![
        hash_line("Weight", &state.weight_commitment),
        hash_line("IO Root", &state.io_root),
        hash_line("Report", &state.report_hash),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  Verifier  ", Style::default().fg(TEXT_DIM)),
            Span::styled(
                truncate_hash(&state.contract, 24),
                Style::default().fg(ACCENT),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Network   ", Style::default().fg(TEXT_DIM)),
            Span::styled(&state.network, Style::default().fg(TEXT)),
        ]),
    ];

    if let Some(count) = state.verification_count {
        lines.push(Line::from(vec![
            Span::styled("  On-chain  ", Style::default().fg(TEXT_DIM)),
            Span::styled(
                format!("{count} verified"),
                Style::default().fg(SUCCESS).bold(),
            ),
        ]));
    }

    let commits = Paragraph::new(lines)
        .block(Block::default()
            .title(Span::styled(" Cryptographic Commitments ", Style::default().fg(HASH_COLOR).bold()))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(HASH_COLOR))
            .style(Style::default().bg(SURFACE))
            .padding(Padding::horizontal(0)));
    frame.render_widget(commits, layout[0]);

    // Tamper detection
    let tamper_lines = vec![
        tamper_line("IO commitment", state.tamper_io),
        tamper_line("Weight commitment", state.tamper_weight),
        tamper_line("Inference output", state.tamper_output),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled("56 adversarial tests", Style::default().fg(SUCCESS)),
            Span::styled(" in test suite", Style::default().fg(TEXT_DIM)),
        ]),
    ];

    let tamper = Paragraph::new(tamper_lines)
        .block(Block::default()
            .title(Span::styled(" Tamper Detection ", Style::default().fg(DANGER).bold()))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(127, 29, 29)))
            .style(Style::default().bg(SURFACE))
            .padding(Padding::new(1, 1, 1, 0)));
    frame.render_widget(tamper, layout[1]);

    // Conversation
    let mut conv_lines: Vec<Line> = Vec::new();
    for (user, ai) in state.turns.iter().rev().take(4) {
        conv_lines.push(Line::from(vec![
            Span::styled("  You ", Style::default().fg(BRAND).bold()),
            Span::styled(truncate_str(user, 40), Style::default().fg(TEXT)),
        ]));
        conv_lines.push(Line::from(vec![
            Span::styled("  AI  ", Style::default().fg(ACCENT).bold()),
            Span::styled(truncate_str(ai, 40), Style::default().fg(TEXT_DIM)),
        ]));
    }

    let conv = Paragraph::new(conv_lines)
        .block(Block::default()
            .title(Span::styled(" Conversation ", Style::default().fg(BRAND).bold()))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BRAND_DIM))
            .style(Style::default().bg(SURFACE))
            .padding(Padding::horizontal(0)));
    frame.render_widget(conv, layout[2]);
}

fn render_footer(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let status = match state.step {
        PipelineStep::Idle => ("READY", TEXT_DIM),
        PipelineStep::Capture => ("CAPTURING", WARN),
        PipelineStep::Prove => ("PROVING", BRAND),
        PipelineStep::Recursive => ("COMPRESSING", ACCENT),
        PipelineStep::Verify => ("VERIFYING", HASH_COLOR),
        PipelineStep::Complete => ("VERIFIED ✓", SUCCESS),
    };

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" ObelyZK ", Style::default().fg(Color::Black).bg(BRAND).bold()),
        Span::raw("  "),
        Span::styled(status.0, Style::default().fg(status.1).bold()),
        Span::raw("  "),
        Span::styled(
            format!("{} turns  {}→{}",
                state.num_turns, state.tokens_in, state.tokens_out),
            Style::default().fg(TEXT_DIM),
        ),
        Span::raw("  "),
        Span::styled("q to exit", Style::default().fg(TEXT_MUTED)),
    ]))
    .block(Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(TEXT_MUTED))
        .style(Style::default().bg(SURFACE)));
    frame.render_widget(footer, area);
}

// ── Helpers ─────────────────────────────────────────────────────────

fn hash_line<'a>(label: &'a str, value: &Option<String>) -> Line<'a> {
    let hash_str = value.as_deref().unwrap_or("pending...");
    Line::from(vec![
        Span::styled(format!("  {label:<8}"), Style::default().fg(TEXT_DIM)),
        Span::styled(truncate_hash(hash_str, 32), Style::default().fg(HASH_COLOR)),
    ])
}

fn tamper_line(name: &str, result: Option<bool>) -> Line<'static> {
    match result {
        Some(true) => Line::from(vec![
            Span::styled("  ✓ ", Style::default().fg(SUCCESS)),
            Span::styled(name.to_string(), Style::default().fg(TEXT)),
            Span::styled(" → ", Style::default().fg(TEXT_MUTED)),
            Span::styled("REJECTED", Style::default().fg(SUCCESS).bold()),
        ]),
        Some(false) => Line::from(vec![
            Span::styled("  ✗ ", Style::default().fg(DANGER)),
            Span::styled(name.to_string(), Style::default().fg(TEXT)),
            Span::styled(" → NOT DETECTED", Style::default().fg(DANGER).bold()),
        ]),
        None => Line::from(vec![
            Span::styled("  · ", Style::default().fg(TEXT_MUTED)),
            Span::styled(name.to_string(), Style::default().fg(TEXT_MUTED)),
        ]),
    }
}

fn truncate_hash(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    format!("{}...{}", &s[..max/2], &s[s.len()-6..])
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max { return s.to_string(); }
    s.chars().take(max - 1).collect::<String>() + "…"
}
