import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
import seaborn as sns

def fillMissing(df):
    filler = df.fillna(0)
    return filler

def processCSV(df):
    missingFill = fillMissing(df)
    return  missingFill
# PLOTS BY REGION
# TOTAL BY STRAND EACH REGION
def totalbyStrandEachRegion(df):
    gr11male = [col for col in df.columns if 'MALE' in col and 'FEMALE' not in col and '11' in col]
    gr12male = [col for col in df.columns if 'MALE' in col and 'FEMALE' not in col and '12' in col]
    gr11female = [col for col in df.columns if 'FEMALE' in col and '11' in col]
    gr12female = [col for col in df.columns if 'FEMALE' in col and '12' in col]
    gr11 = [col for col in df.columns if '11' in col]
    gr12 = [col for col in df.columns if '12' in col]
    grandtotal = gr11 + gr12
    male = [col for col in df.columns if 'MALE' in col and 'FEMALE' not in col]
    sortedmale = sorted(male, key=lambda x: (x.split(' ')[1], x.split(' ')[0]))
    groupedmale = df[sortedmale].T.groupby(lambda x: x.split(' ')[1]).sum().T
    groupedmale['Year'] = df['Year']
    groupedmale['Sector'] = df['Sector']
    groupedmale['Region'] = df['Region']
    female = [col for col in df.columns if 'FEMALE' in col]
    sortedfemale = sorted(female, key=lambda x: (x.split(' ')[1], x.split(' ')[0]))
    groupedfemale = df[sortedfemale].T.groupby(lambda x: x.split(' ')[1]).sum().T
    groupedfemale['Year'] = df['Year']
    groupedfemale['Sector'] = df['Sector']
    groupedfemale['Region'] = df['Region']
    regions = df['Region'].unique()
    sectors = df['Sector'].unique()
    allmandf = male + female
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.2)
    region_index = [0]
    gradegender_index = [0]
    sector_index = [0]
    axprevreg = plt.axes([0.1, 0.05, 0.1, 0.075])
    axnextreg = plt.axes([0.2, 0.05, 0.1, 0.075])
    axprevsect = plt.axes([0.4, 0.05, 0.1, 0.075])
    axnextsect = plt.axes([0.5, 0.05, 0.1, 0.075])
    axprevselect = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnextselect = plt.axes([0.8, 0.05, 0.1, 0.075])
    axnextregion = Button(axnextreg, 'Next Region')
    axpreviousregion = Button(axprevreg, 'Previous Region')
    axnextsector = Button(axnextsect, 'Next Sector')
    axprevioussector = Button(axprevsect, 'Previous Sector')
    axnextselection = Button(axnextselect, 'Next Grade/Gender')
    axpreviousselection = Button(axprevselect, 'Previous Grade/Gender')
    def update_plot(region_idx, selection_idx, sector_idx):
        if 15 <= selection_idx <= 22:
            axnextsector.set_active(False)
            axprevioussector.set_active(False)
        else:
            axnextsector.set_active(True)
            axprevioussector.set_active(True)
        ax.clear()
        current_region = regions[region_idx]
        current_sector = sectors[sector_idx]
        gradegender = {
            0: (gr11male, "Grade 11 Male Students"),#para sa male students by strand, sector,region
            1: (gr11female, "Grade 11 Female Students"),#female students by strand, sector,region
            2: (gr12male, "Grade 12 Male Students"),#para sa male students by strand, sector,region
            3: (gr12female, "Grade 12 Female Students"),#para sa female students by strand, sector,region
            4: (groupedmale.columns.tolist(), "Grade 11 and 12 Male Students"),#group by strand 11abm + 12abm ..etc male
            5: (groupedfemale.columns.tolist(), "Grade 11 and 12 Female Students"),#group by strand 11abm + 12abm ..etc female
            6: (gr11male, "Total Grade 11 Male Students Grand Total"),#total grade 11 by sector, region male
            7: (gr11female, "Total Grade 11 Female Students Grand Total"),#total grade 11 by sector, region female
            8: (gr12male, "Total Grade 12 Male Students Grand Total"),#total grade 12 by sector, region male
            9: (gr12female, "Total Grade 12 Female Students Grand Total"),#total grade 12 by sector, region female
            10: (gr11, "Grade 11 Grand Total"),#grandtotal gr11
            11: (gr12, "Grade 12 Grand Total"),#grandtotal gr12
            12: (grandtotal, "Grade 11 and Grade 12"),#grandtotal gr11+gr12
            13: (male, "Grand Total Male"),#grandtotal male gr11+gr12
            14: (female, "Grand Total Female"),#grandtotal female gr11+gr12
            15: (male, "Total Grade 11 and 12 Male"),#total male 11 and 12 all sector by strand
            16: (female, "Total Grade 12 and 12 Female"),#total female 11 and 12 all sector by strand
            17: (groupedmale.columns.tolist(), "Total 11 and 12 Male Students"),#total male 11 and 12 all sector and merge strand 11abm+12abm ..etc
            18: (groupedfemale.columns.tolist(), "Total 11 and 12 Female Students"),#total female 11 and 12 all sector and merge strand 11abm+12abm ..etc
            19: (male, "OVERALL MALE"),#overallmale all sector
            20: (female,"OVERALL FEMALE"),#overallfemale all sector
            21: (allmandf, "OVERALL MALE AND FEMALE"),#overmale + overfemale
            22: (df, "TOTAL STUDENT OF EACH REGION OF ALL SECTORS IN YEARS 2016 - 2021")#total student all sector by region
        }
        selected_gradegender, title = gradegender[selection_idx]
        print(title)
        colors = sns.color_palette("tab20", n_colors=20)
        if selection_idx in [0, 1, 2, 3]:
            region_sector_data = df[(df['Region'] == current_region) & (df['Sector'] == current_sector)]
            region_data = region_sector_data.groupby('Year')[selected_gradegender].sum()
            for i, strand in enumerate(selected_gradegender):
                ax.plot(region_data.index, region_data[strand], label=f'{strand}', color=colors[i], marker='o',
                        markersize=5)
            ax.set_title(f"{title} enrolled in different strands in {current_region} - Sector {current_sector}")
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of students')
            ax.legend()
        elif selection_idx in [4, 5]:
            if selection_idx == 4:
                region_data = groupedmale[
                    (groupedmale['Region'] == current_region) & (groupedmale['Sector'] == current_sector)]
            else:
                region_data = groupedfemale[
                    (groupedfemale['Region'] == current_region) & (groupedfemale['Sector'] == current_sector)]
            region_data_year = region_data.groupby('Year')[selected_gradegender].sum()
            strand_columns = [col for col in region_data_year.columns if col not in ['Region', 'Year', 'Sector']]
            for i, strand in enumerate(strand_columns):
                ax.plot(region_data_year.index, region_data_year[strand], label=f'{strand}', marker='o', markersize=5, color=colors[i])
            ax.set_title(f"{title} enrolled in different strands in {current_region} - Sector {current_sector}")
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of students')
            ax.legend()
        elif selection_idx in [15,16]:
            region_sector_datas = df[(df['Region'] == current_region)]
            region_datas = region_sector_datas.groupby('Year')[selected_gradegender].sum()
            for i, strand in enumerate(selected_gradegender):
                ax.plot(region_datas.index, region_datas[strand], label=f'{strand}', marker='o', markersize=5, color=colors[i])
            ax.set_title(f"{title} enrolled in {current_region} - ALL SECTORS")
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of students')
            ax.legend()
        elif selection_idx in [17,18]:
            if selection_idx == 17:
                region_data = groupedmale[
                    (groupedmale['Region'] == current_region)]
            else:
                region_data = groupedfemale[
                    (groupedfemale['Region'] == current_region)]
            region_data_year = region_data.groupby('Year')[selected_gradegender].sum()
            strand_columns = [col for col in region_data_year.columns if col not in ['Region', 'Year', 'Sector']]
            for i, strand in enumerate(strand_columns):
                ax.plot(region_data_year.index, region_data_year[strand], label=f'{strand}', marker='o', markersize=5, color=colors[i])
            ax.set_title(f"{title} enrolled in {current_region} - ALL SECTORS")
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of students')
            ax.legend()
        elif selection_idx in [19,20,21]:
            region_sector_data = df[(df['Region'] == current_region)]
            region_datas = region_sector_data.groupby('Year')[selected_gradegender].sum().sum(axis=1)
            # ax.bar(region_datas.index, region_datas.values, label=title)
            ax.bar(region_datas.index, region_datas.values, label=title, color=colors)
            ax.set_title(f"{title} enrolled in {current_region} - ALL SECTORS")
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of students')
            ax.legend()
        elif selection_idx in [22]:
            numeric_df = df.select_dtypes(include='number')
            result = df.groupby('Region')[numeric_df.columns].sum()
            result['Total Students'] = result.sum(axis=1)
            regionss = result.index.str.split(' - ').str[0]
            total_students = result['Total Students']
            ax.barh(regionss, total_students, color=colors)
            ax.set_title('Total Students by Region (2016-2021)')
            ax.set_xlabel('Student Count')
            ax.set_ylabel('Regions')
        else:
            region_sector_data = df[(df['Region'] == current_region) & (df['Sector'] == current_sector)]
            region_datas = region_sector_data.groupby('Year')[selected_gradegender].sum().sum(axis=1)
            ax.bar(region_datas.index, region_datas.values, label=title, color=colors)
            ax.set_title(f"{title} enrolled in {current_region} - Sector {current_sector}")
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of students')
            ax.legend()
        plt.grid(True)
        plt.draw()

    def next_region(event):
        region_index[0] = (region_index[0] + 1) % len(regions)
        gradegender_index[0] = 0
        print(region_index)
        update_plot(region_index[0], gradegender_index[0], sector_index[0])
    def previous_region(event):
        region_index[0] = (region_index[0] - 1) % len(regions)
        gradegender_index[0] = 0
        update_plot(region_index[0], gradegender_index[0], sector_index[0])
    def next_gradegender(event):
        if gradegender_index[0] < 22:
            gradegender_index[0] += 1
        else:
            gradegender_index[0] = 0
        update_plot(region_index[0], gradegender_index[0], sector_index[0])
    def previous_gradegender(event):
        if gradegender_index[0] > 0:
            gradegender_index[0] -= 1
        else:
            gradegender_index[0] = 22
        update_plot(region_index[0], gradegender_index[0], sector_index[0])

    def next_sector(event):
        sector_index[0] = (sector_index[0] + 1) % len(sectors)
        gradegender_index[0] = 0
        update_plot(region_index[0], gradegender_index[0], sector_index[0])
    def previous_sector(event):
        sector_index[0] = (sector_index[0] - 1) % len(sectors)
        gradegender_index[0] = 0
        update_plot(region_index[0], gradegender_index[0], sector_index[0])
    axnextregion.on_clicked(next_region)
    axpreviousregion.on_clicked(previous_region)
    axnextselection.on_clicked(next_gradegender)
    axpreviousselection.on_clicked(previous_gradegender)
    axnextsector.on_clicked(next_sector)
    axprevioussector.on_clicked(previous_sector)
    update_plot(region_index[0], gradegender_index[0], sector_index[0])
    plt.show()
# PIE BY SECTOR
def totalStudentsbySector(df):
    male = [col for col in df.columns if 'MALE' in col and 'FEMALE' not in col]
    female = [col for col in df.columns if 'FEMALE' in col]
    regions = df['Region'].unique()
    years = df['Year'].unique()
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.3)
    axprev_region = plt.axes([0.1, 0.05, 0.1, 0.075])
    axnext_region = plt.axes([0.2, 0.05, 0.1, 0.075])
    axprev_chart = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext_chart = plt.axes([0.8, 0.05, 0.1, 0.075])
    axprev_gender = plt.axes([0.4, 0.05, 0.1, 0.075])
    axnext_gender = plt.axes([0.5, 0.05, 0.1, 0.075])
    bnext_region = Button(axnext_region, 'Next Region')
    bprev_region = Button(axprev_region, 'Previous Region')
    bnext_chart = Button(axnext_chart, 'Next Chart')
    bprev_chart = Button(axprev_chart, 'Previous Chart')
    bnext_gender = Button(axnext_gender, 'Next Gender')
    bprev_gender = Button(axprev_gender, 'Previous Gender')
    current_region_index = [0]
    current_year_index = [0]
    current_gender_index = [0]
    def update_plot(selection_idx):
        print(selection_idx)
        gender = {
            0: (male, "Male"),
            1: (female, "Female"),
            2: (df.columns, "Overall")
        }
        ax.clear()
        current_region = regions[current_region_index[0]]
        current_year = years[current_year_index[0]]
        current_gender = gender[current_gender_index[0]]
        gender_cols = current_gender[0]
        gender_label = current_gender[1]
        if selection_idx in [0,1]:
            by_sector = df.groupby(['Region', 'Year', 'Sector'])[gender_cols].sum()
            region_year_data = by_sector.loc[(current_region, current_year)]
            sector_counts = region_year_data.sum(axis=1)
            print(sector_counts)
            ax.pie(
                sector_counts,
                labels=sector_counts.index,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                colors=sns.color_palette("rocket_r"),
                radius=1.2
            )
            ax.set_title(f"{gender_label} Students in ({current_year}) by Sector", pad=20, loc='center', y=-0.2)
        else:
            by_sector = df.groupby(['Region', 'Year', 'Sector']).sum()
            region_year_data = by_sector.loc[(current_region, current_year)]
            sector_counts = region_year_data.sum(axis=1)
            print(sector_counts)
            ax.pie(
                sector_counts,
                labels=sector_counts.index,
                autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                colors=sns.color_palette("rocket_r"),
                radius=1.2
            )
            ax.set_title(f"{gender_label} Students in ({current_year}) by Sector",pad=20, loc='center', y=-0.2)

        fig.suptitle(f'{gender_label} Students Percentage for {current_region}')
        plt.draw()
    def next_region(event):
        if current_region_index[0] < len(regions) - 1:
            current_region_index[0] += 1
        else:
            current_region_index[0] = 0
        current_year_index[0] = 0
        update_plot(current_gender_index[0])
    def prev_region(event):
        if current_region_index[0] > 0:
            current_region_index[0] -= 1
        else:
            current_region_index[0] = len(regions) - 1
        current_year_index[0] = 0
        update_plot(current_gender_index[0])
    def next_chart(event):
        if current_year_index[0] < len(years) - 1:
            current_year_index[0] += 1
        else:
            current_year_index[0] = 0
        update_plot(current_gender_index[0])
    def prev_chart(event):
        if current_year_index[0] > 0:
            current_year_index[0] -= 1
        else:
            current_year_index[0] = len(years) - 1
        update_plot(current_gender_index[0])
    def next_gender(event):
        if current_gender_index[0] < 2:
            current_gender_index[0] += 1
        else:
            current_gender_index[0] = 0
        update_plot(current_gender_index[0])
    def prev_gender(event):
        if current_gender_index[0] > 0:
            current_gender_index[0] -= 1
        else:
            current_gender_index[0] = 2
        update_plot(current_gender_index[0])
    bnext_region.on_clicked(next_region)
    bprev_region.on_clicked(prev_region)
    bnext_chart.on_clicked(next_chart)
    bprev_chart.on_clicked(prev_chart)
    bnext_gender.on_clicked(next_gender)
    bprev_gender.on_clicked(prev_gender)
    update_plot(current_gender_index[0])
    plt.show()
# PIE BY REGION
def totalStudentsbyRegion(df):
    allmandf = [col for col in df.columns if 'MALE' in col]
    years = df['Year'].unique()
    regions = df['Region'].unique()
    current_year_index = [0]
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.3)
    axprev_chart = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext_chart = plt.axes([0.8, 0.05, 0.1, 0.075])
    bnext_chart = Button(axnext_chart, 'Next Year')
    bprev_chart = Button(axprev_chart, 'Previous Year')
    def update_plot():
        ax.clear()
        current_year = years[current_year_index[0]]
        df_filtered = df[df['Year'] == current_year]
        total_counts = df_filtered.groupby('Region')[allmandf].sum().sum(axis=1)
        num_regions = len(regions)
        color = sns.color_palette("flare", n_colors=num_regions)

        ax.pie(total_counts, labels=regions, autopct='%1.1f%%', radius=1.5,
               colors=color)
        ax.set_title(f'Total Students - Year {current_year}', pad=20, loc='center', y=-0.4)
        plt.draw()
    def next_chart(event):
        if current_year_index[0] < len(years) - 1:
            current_year_index[0] += 1
        else:
            current_year_index[0] = 0
        update_plot()
    def prev_chart(event):
        if current_year_index[0] > 0:
            current_year_index[0] -= 1
        else:
            current_year_index[0] = len(years) - 1
        update_plot()
    bnext_chart.on_clicked(next_chart)
    bprev_chart.on_clicked(prev_chart)
    update_plot()
    plt.show()
# Aggregate and export each by region
def aggregateregion(df):
    regions = df['Region'].unique()
    for region in regions:
        region_sector_data = df[df['Region'] == region]
        region_data = region_sector_data.groupby('Year').sum()
        filename = f"Aggregated Data in {region} from year 2016 - 2021.csv"
        region_data.to_csv(filename)
# Aggregate tanan
def aggregateall(df):
    all_regions_data = []
    regions = df['Region'].unique()
    for region in regions:
        region_sector_data = df[df['Region'] == region]
        region_data = region_sector_data.groupby('Year').sum()
        region_data['Region'] = region
        all_regions_data.append(region_data)
    all_data = pd.concat(all_regions_data)
    all_data.to_csv('Aggregated Data from year 2016 - 2021.csv')
#aggregate all from 2016-2021 by region all sectors
def agg20162021(df):
    numeric_df = df.select_dtypes(include='number')
    result = df.groupby('Region')[numeric_df.columns].sum()
    result['Total Students'] = result.sum(axis=1)
    result[['Total Students']].to_csv('Total Students of Each Region of all Sectors from 2016 - 2021.csv', index=True)











