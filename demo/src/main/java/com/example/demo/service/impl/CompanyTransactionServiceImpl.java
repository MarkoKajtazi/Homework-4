package com.example.demo.service.impl;

import com.example.demo.model.CompanyTransaction;
import com.example.demo.repository.CompanyTransactionRepository;
import com.example.demo.service.CompanyTransactionService;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.hibernate.annotations.Fetch;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Service
public class CompanyTransactionServiceImpl implements CompanyTransactionService {

    private final RestTemplate restTemplate = new RestTemplate();

    @Autowired
    private CompanyTransactionRepository companyTransactionRepository;

    @Override
    public List<CompanyTransaction> getAllData() {
        return companyTransactionRepository.findAll();
    }

    @Override
    public CompanyTransaction getDataById(Long id) {
        return companyTransactionRepository.findById(id).orElse(null);
    }

    @Override
    public List<String> getPrediction(String companyId) {
        String filePath = String.format("/usr/src/app/src/main/resources/predicted_data_frame_%s.csv", companyId);
        try (CSVParser parser = CSVFormat.DEFAULT.withHeader().parse(new FileReader(filePath))) {
            List<CSVRecord> records = parser.getRecords();
            if (!records.isEmpty()) {
                CSVRecord firstDay = records.get(records.size() - 2);
                CSVRecord secondDay = records.get(records.size() - 1);
                List<String> predictions = new ArrayList<>();
                predictions.add(firstDay.get("Predicted_Price").toString());
                predictions.add(secondDay.get("Predicted_Price").toString());
                return predictions;
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }
        return null;
    }


}
