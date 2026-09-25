import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import thucia.core.cases as cases
import thucia.core.geo as geo
from thucia.core.fs import read_db
from thucia.core.fs import write_db

PROJECTS_ROOT = Path(__file__).parent.parent.parent.resolve() / "projects"


def launch_dashboard():
    app_path = Path(__file__).parent.parent / "viz" / "dashboard" / "app.py"
    subprocess.run(["streamlit", "run", app_path], check=True)


def cases_per_month(
    project: str = "cases",
    cases_col: str = "Cases",
    cases_file: str = "cases",
    output_file: str = "cases_per_month",
    projects_root: Path | None = None,
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases data
    tdf = read_db(path / cases_file)

    # Aggregate cases per month
    tdf = cases.cases_per_month(tdf, cases_col=cases_col)
    tdf = geo.ensure_all_regions(
        tdf
    )  # Ensure all Admin-2 regions included for covariate maps

    # Write output
    write_db(tdf, path / output_file)


def plot_cases_per_month(
    project: str = "cases",
    cases_file: str = "cases_per_month",
    projects_root: Path | None = None,
    title: str | None = None,
    date_col: str = "Date",
    cases_col: str = "Cases",
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases per month data
    tdf = read_db(path / cases_file)

    # Aggregate and plot
    df = tdf[[date_col, cases_col]].groupby([date_col])[cases_col].sum()
    df.index = df.index.to_timestamp(how="end")
    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.title(title or f"Monthly Cases for {project}")
    plt.xlabel(date_col)
    plt.ylabel(cases_col)
    plt.show()


def cases_per_week(
    project: str = "cases",
    cases_col: str = "Cases",
    cases_file: str = "cases",
    output_file: str = "cases_per_week",
    projects_root: Path | None = None,
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases data
    tdf = read_db(path / cases_file)

    # Aggregate cases per week
    tdf = cases.aggregate_cases(tdf, cases_col=cases_col, freq="W")

    # Write output
    write_db(tdf, path / output_file)


def plot_cases_per_week(
    project: str = "cases",
    cases_file: str = "cases_per_week",
    projects_root: Path | None = None,
    title: str | None = None,
    date_col: str = "Date",
    cases_col: str = "Cases",
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases per week data
    tdf = read_db(path / cases_file)

    # Aggregate and plot
    df = tdf[[date_col, cases_col]].groupby([date_col])[cases_col].sum()
    df.index = df.index.to_timestamp(how="end")
    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.title(title or f"Weekly Cases for {project}")
    plt.xlabel(date_col)
    plt.ylabel(cases_col)
    plt.show()


def cases_per_day(
    project: str = "cases",
    cases_col: str = "Cases",
    cases_file: str = "cases",
    output_file: str = "cases_per_day",
    projects_root: Path | None = None,
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases data
    tdf = read_db(path / cases_file)

    # Aggregate cases per day
    tdf = cases.aggregate_cases(tdf, cases_col=cases_col, freq="D")

    # Write output
    write_db(tdf, path / output_file)


def plot_cases_per_day(
    project: str = "cases",
    cases_file: str = "cases_per_day",
    projects_root: Path | None = None,
    title: str | None = None,
    date_col: str = "Date",
    cases_col: str = "Cases",
):
    # If project specified, use projects root, otherwise use current directory
    projects_root = projects_root or PROJECTS_ROOT
    path = (Path(projects_root) / project).resolve() if project else Path(".").resolve()

    # Read cases per day data
    tdf = read_db(path / cases_file)

    # Aggregate and plot
    df = tdf[[date_col, cases_col]].groupby([date_col])[cases_col].sum()
    df.index = df.index.to_timestamp(how="end")
    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.title(title or f"Daily Cases for {project}")
    plt.xlabel(date_col)
    plt.ylabel(cases_col)
    plt.show()
