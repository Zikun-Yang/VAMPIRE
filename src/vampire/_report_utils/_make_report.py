def make_report(job_dir: str, template_html_path: str, data: dict) -> None:
    """
    make report
    Input:
        job_dir: str
        template_html_path: str
        data: dict
    Output:
        None
    """
    with open(template_html_path, 'r') as f:
        template = f.read()
    for key, value in data.items():
        template = template.replace(f"${key}$", str(value))
    with open(f"{job_dir}/web_summary.html", 'w') as f:
        f.write(template)



