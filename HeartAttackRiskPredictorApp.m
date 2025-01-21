function HeartAttackRiskPredictorApp()
    % Main App Window
    app = uifigure('Name', 'Heart Attack Risk Predictor', ...
                   'Position', [500, 300, 400, 400]);

    % Labels and Input Fields
    uilabel(app, 'Text', 'Age:', 'Position', [20, 330, 100, 30]);
    ageInput = uieditfield(app, 'numeric', 'Position', [120, 330, 200, 30]);

    uilabel(app, 'Text', 'Cholesterol:', 'Position', [20, 290, 100, 30]);
    cholesterolInput = uieditfield(app, 'numeric', 'Position', [120, 290, 200, 30]);

    uilabel(app, 'Text', 'Blood Pressure (e.g., 120/80):', ...
            'Position', [20, 250, 150, 30]);
    bpInput = uieditfield(app, 'text', 'Position', [170, 250, 150, 30]);

    uilabel(app, 'Text', 'Sex:', 'Position', [20, 210, 100, 30]);
    sexDropdown = uidropdown(app, 'Items', {'Male', 'Female'}, ...
                             'Position', [120, 210, 200, 30]);

    uilabel(app, 'Text', 'BMI:', 'Position', [20, 170, 100, 30]);
    bmiInput = uieditfield(app, 'numeric', 'Position', [120, 170, 200, 30]);

    % Compute Button
    computeRiskButton = uibutton(app, 'Text', 'Compute Risk', ...
                                 'Position', [150, 100, 100, 30], ...
                                 'ButtonPushedFcn', @(btn, event) computeRisk(ageInput, cholesterolInput, bpInput, sexDropdown, bmiInput, app));
end

function computeRisk(ageInput, cholesterolInput, bpInput, sexDropdown, bmiInput, app)
    % Step 1: Load the Trained FIS Model
    try
        if isfile('CI_Assignment_trainedFIS.mat')
            load('CI_Assignment_trainedFIS.mat', 'trainedFis');
        else
            uialert(app, 'Trained FIS file not found. Ensure the file is in the correct directory.', 'File Error');
            return;
        end
    catch
        uialert(app, 'Error loading FIS. Check the file content and variable names.', 'Load Error');
        return;
    end

    % Step 2: Retrieve User Inputs
    try
        age = ageInput.Value;
        cholesterol = cholesterolInput.Value;
        bp = bpInput.Value;
        sex = sexDropdown.Value;
        bmi = bmiInput.Value;

        if isempty(age) || isempty(cholesterol) || isempty(bp) || isempty(sex) || isempty(bmi)
            uialert(app, 'Please fill in all fields.', 'Input Error');
            return;
        end

        % Step 3: Parse Blood Pressure
        bpSplit = strsplit(bp, '/');
        if length(bpSplit) ~= 2
            uialert(app, 'Blood pressure must be in the format "Systolic/Diastolic".', 'Input Error');
            return;
        end
        systolic = str2double(bpSplit{1});
        diastolic = str2double(bpSplit{2});
        if isnan(systolic) || isnan(diastolic)
            uialert(app, 'Invalid blood pressure values.', 'Input Error');
            return;
        end

        % Calculate Mean Arterial Pressure (MAP)
        map = (2 * diastolic + systolic) / 3;

        % Step 4: Encode Inputs
        sexEncoded = strcmp(sex, 'Male'); % 1 for Male, 0 for Female

        % Step 5: Create Input Vector
        % Ensure the inputs are in the correct order expected by the FIS
        inputVector = [age, cholesterol, map, sexEncoded, bmi];
        if length(inputVector) ~= length(trainedFis.Inputs)
            uialert(app, sprintf('FIS expects %d inputs, but provided %d.', ...
                   length(trainedFis.Inputs), length(inputVector)), 'Input Error');
            return;
        end

        % Step 6: Predict Heart Attack Risk
        riskScore = evalfis(trainedFis, inputVector);

        % Display the Predicted Risk Score
        uialert(app, sprintf('Predicted Heart Attack Risk: %.2f%%', riskScore * 100), 'Prediction Result');

    catch ME
        % Handle Unexpected Errors
        uialert(app, sprintf('An error occurred: %s', ME.message), 'Error');
    end
end
