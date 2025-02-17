
def plot_animation(df):
    brush  = alt.selection_interval ()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
        ).properties(
            width=414,
            height=250
        ).add_selection(
            brush
        ).interactive()
    
    figure = chart1.encode(y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude'))) | chart1.encode(y ='amplitude after processing').add_selection(
            brush)
    return figure


def Dynamic_graph(signal_x_axis, signal_y_axis, signal_y_axis1,start_btn,pause_btn,resume_btn):
        df = pd.DataFrame({'time': signal_x_axis[::200], 'amplitude': signal_y_axis[:: 200], 'amplitude after processing': signal_y_axis1[::200]}, columns=['time', 'amplitude','amplitude after processing'])

        lines = plot_animation(df)
        line_plot = st.altair_chart(lines)

        N = df.shape[0]  # number of elements in the dataframe
        burst = 10       # number of elements  to add to the plot
        size = burst     # size of the current dataset

        if start_btn:
            for i in range(1, N):
                variabls.start=i
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                variabls.graph_size=size
                size = i * burst 

        if resume_btn: 
            for i in range( variabls.start,N):
                variabls.start=i
                step_df     = df.iloc[0:size]
                lines       = plot_animation(step_df)
                line_plot   = line_plot.altair_chart(lines)
                variabls.graph_size=size
                size = i * burst

        if pause_btn:
            step_df = df.iloc[0:variabls.graph_size]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)

